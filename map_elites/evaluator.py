import torch
import torch.nn as nn
import time
import cloudpickle
import pickle
import numpy as np
from faster_fifo import Queue
from torch.multiprocessing import Event, Pipe, Process as TorchProcess
from utils.vectorized import BatchMLP
from map_elites import common as cm
from itertools import count
from collections import deque


from utils.logger import log
from utils.utils import get_least_busy_gpu
from pynvml import *



# flags for the archive
UNUSED = 0
MAPPED = 1


class Individual(object):
    _ids = count(0)

    def __init__(self, genotype, phenotype, fitness, centroid=None):
        """
        A single agent
        param genotype:  The parameters that produced the behavior. I.e. neural network, etc.
        param phenotype: the resultant behavior i.e. the behavioral descriptor
        param fitness: the fitness of the model. In the case of a neural network, this is the total accumulated rewards
        param centroid: the closest CVT generator (a behavior) to the behavior that this individual exhibits
        """
        genotype.id = next(self._ids)
        Individual.current_id = genotype.id  # TODO: not sure what this is for
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness
        self.centroid = centroid
        self.novelty = None



class Evaluator(object):
    def __init__(self, env_fns, all_actors, eval_cache, eval_cache_locks, elites_map, batch_size, seed, num_parallel,
                 actors_per_worker, num_gpus, kdt, msgr_remote, eval_in_queue):
        '''
        A class for parallel evaluations
        :param env_fns: 2d list of gym envs (num_processes x env_batch_size). Each worker gets a batch of envs to step through
        :param batch_size:
        :param seed:
        :param num_parallel:
        :param actors_per_worker:
        :param num_gpus:
        '''
        self.num_processes = num_parallel * num_gpus if num_gpus > 1 else num_parallel
        self.num_gpus = num_gpus
        self.eval_in_queue = eval_in_queue
        self.eval_out_queue = Queue(max_size_bytes=int(1e7))
        self.remotes, self.locals = zip(*[Pipe() for _ in range(self.num_processes)])

        # logging variables
        self.avg_stats_intervals = (2, 12, 60)  # 10 seconds, 1 minute, 5 minutes
        self._report_interval = 5.0  # report every 5 seconds
        self._last_report = 0.0
        self.eval_stats = deque([], maxlen=max(self.avg_stats_intervals))

        self.processes = [EvalWorker(process_id,
                                     env_fn,
                                     all_actors,
                                     eval_cache,
                                     eval_cache_locks,
                                     elites_map,
                                     batch_size,
                                     seed,
                                     actors_per_worker,
                                     num_gpus,
                                     self.eval_in_queue,
                                     self.eval_out_queue,
                                     self.remotes[process_id],
                                     kdt,
                                     msgr_remote) for process_id, env_fn in enumerate(env_fns)]

    @property
    def report_interval(self):
        return self._report_interval

    @property
    def last_report(self):
        return self._last_report

    def close_envs(self):
        for process in self.processes:
            process.terminate()

    def report_evals(self, total_env_steps, total_evals):
        now = time.time()
        self.eval_stats.append((now, total_env_steps, total_evals))
        if len(self.eval_stats) <= 1: return 0.0, 0.0

        fps, eps = [], []
        for avg_interval in self.avg_stats_intervals:
            past_time, past_frames, past_evals = self.eval_stats[max(0, len(self.eval_stats) - 1 - avg_interval)]
            fps.append(round((total_env_steps - past_frames) / (now - past_time), 1))
            eps.append(round((total_evals - past_evals) / (now - past_time), 1))

        log.debug(f'Evals/sec (EPS) 10 sec: {eps[0]}, 60 sec: {eps[1]}, 300 sec: {eps[2]}, '
                  f'FPS 10 sec: {fps[0]}, 60 sec: {fps[1]}, 300 sec: {fps[2]}')

        self._last_report = now
        return fps[2], eps[2]



class EvalWorker(object):
    def __init__(self, process_id, env_fns, all_actors, eval_cache, eval_cache_locks, elites_map, batch_size, master_seed,
                 actors_per_worker, num_gpus, eval_in_queue, eval_out_queue, remote, kdt, msgr_remote):
        self.env_fn_wrappers = env_fns
        self.all_actors = all_actors
        self.eval_cache = eval_cache
        self.eval_cache_locks = eval_cache_locks
        self.elites_map = elites_map
        self.batch_size = batch_size
        self.master_seed = master_seed
        self.actors_per_worker = actors_per_worker
        self.num_gpus = num_gpus
        self._terminate = False
        self.eval_in_queue = eval_in_queue
        self.eval_out_queue = eval_out_queue
        self.remote = remote
        self.msgr_remote = msgr_remote
        self.kdt = kdt  # centroidal voronoi tessalations

        self.process = TorchProcess(target=self._run, daemon=True)
        self.process.start()

    def _run(self):
        if self.num_gpus:
            nvmlInit()  # for tracking available gpu resources

        # start the simulations
        envs = [self.env_fn_wrappers[i]() for i in range(len(self.env_fn_wrappers))]

        # begin the process loop
        while not self._terminate:
            try:
                eval_id, evolved_actor_keys = self.eval_in_queue.get(block=True, timeout=1e9)  # TODO: make sure this interface is implemented correctly
                start_time = time.time()
                actors = []
                for key in evolved_actor_keys:
                    self.eval_cache_locks[key].acquire()
                    actors += [self.eval_cache[key]]
                    self.eval_cache_locks[key].release()
                assert len(envs) % len(actors) == 0, f'Number of envs should be a multiple of the number of policies. ' \
                                                     f'Got {len(envs)} envs and {len(actors)} policies'
                gpu_id = get_least_busy_gpu(self.num_gpus) if self.num_gpus else None
                device = torch.device(f'cuda:{gpu_id}' if (torch.cuda.is_available() and gpu_id is not None) else 'cpu')
                batch_actors = BatchMLP(actors, device)
                num_actors = len(actors)
                for env in envs:
                    env.seed(int((self.master_seed * 100) * eval_id))

                obs = torch.from_numpy(np.array([env.reset() for env in envs])).reshape(num_actors, -1).to(device)
                rews = [0 for _ in range(num_actors)]
                dones = [False for _ in range(num_actors)]
                infos = [None for _ in range(num_actors)]
                # get a batch of trajectories and rewards
                while not all(dones):
                    obs_arr = []
                    with torch.no_grad():
                        acts = batch_actors(obs).cpu().detach().numpy()
                        for idx, (act, env) in enumerate(zip(acts, envs)):
                            obs, rew, done, info = env.step(act)
                            rews[idx] += rew * (1 - dones[idx])
                            obs_arr.append(obs)
                            dones[idx] = done
                            infos[idx] = info
                        obs = torch.from_numpy(np.array(obs_arr)).reshape(num_actors, -1).to(device)
                ep_lengths = [env.ep_length for env in envs]
                frames = sum(ep_lengths)
                bds = [info['desc'] for info in infos]  # list of behavioral descriptors
                # res = [[rew, ep_len, bd] for rew, ep_len, bd in zip(rews, ep_lengths, bds)]

                runtime = time.time() - start_time
                self._map_agents(actors, bds, rews, runtime, frames)
            except BaseException:
                pass

        for env in envs:
            env.close()

    def terminate(self):
        self._terminate = True

    def _map_agents(self, actors, descs, rews, runtime, frames):
        '''
        Map the evaluated agents using their behavior descriptors. Send the metadata back to the main process for logging
        :param behav_descs: behavior descriptors of a batch of evaluated agents
        :param rews: fitness scores of a batch of evaluated agents
        '''
        metadata = []
        for actor, desc, rew in zip(actors, descs, rews):
            added = False
            agent = Individual(genotype=actor, phenotype=desc, fitness=rew)
            niche_index = self.kdt.query([desc], k=1)[1][0][0]  # get the closest voronoi cell to the behavior descriptor
            niche = self.kdt.data[niche_index]
            n = cm.make_hashable(niche)
            agent.centroid = n
            if n in self.elites_map:
                # natural selection
                map_agent_id, fitness = self.elites_map[n]  # TODO: make sure this interface is correctly implemented
                if agent.fitness > fitness:
                    self.elites_map[n] = (map_agent_id, agent.fitness)
                    # override the existing agent in the actors pool @ map_agent_id. This species goes extinct b/c a more fit one was found
                    self.all_actors[map_agent_id] = (agent, MAPPED)
                    added = True
            else:
                # need to find a new, unused agent id since this agent maps to a new cell
                agent_id = self._find_available_agent_id()
                self.elites_map[n] = (agent_id, agent.fitness)
                self.all_actors[agent_id] = (agent, MAPPED)
                added = True

            if added:
                md = (agent.genotype.id, agent.fitness, str(agent.phenotype).strip("[]"), str(agent.centroid).strip("()"),
                      agent.genotype.parent_1_id, agent.genotype.parent_2_id, agent.genotype.type, agent.genotype.novel, agent.genotype.delta_f)
                metadata.append(md)

        self.msgr_remote.send((metadata, runtime, frames))

    def _find_available_agent_id(self):
        '''
        If an agent maps to a new cell in the elites map, need to give it a new, unused id from the pool of agents
        :returns an agent id that's not being used by some other agent in the elites map
        '''
        return next(i for i in range(len(self.all_actors)) if self.all_actors[i][1] == UNUSED)



