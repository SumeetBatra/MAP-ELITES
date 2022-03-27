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
from map_elites.cvt import Individual

from torch import Tensor

from utils.logger import log
from utils.utils import get_least_busy_gpu
from pynvml import *



class Evaluator(object):
    def __init__(self, env_fns, batch_size, seed, num_parallel, actors_per_worker, num_gpus):
        '''
        A class for parallel evaluations
        :param env_fns: 2d list of gym envs. Each worker gets a batch of envs to step through
        :param batch_size:
        :param seed:
        :param num_parallel:
        :param actors_per_worker:
        :param num_gpus:
        '''
        self.num_processes = num_parallel
        self.num_gpus = num_gpus
        self.eval_in_queue = Queue(max_size_bytes=int(1e7))
        self.eval_out_queue = Queue(max_size_bytes=int(1e7))


class EvalWorker(object):
    def __init__(self, env_fns, all_actors, all_evolved_actors, elites_map, batch_size, master_seed, num_parallel,
                 actors_per_worker, num_gpus, eval_in_queue, eval_out_queue, remote, kdt):
        self.env_fn_wrappers = env_fns
        self.all_actors = all_actors
        self.all_evolved_actors = all_evolved_actors
        self.elites_map = elites_map
        self.batch_size = batch_size
        self.master_seed = master_seed
        self.num_parallel = num_parallel
        self.actors_per_worker = actors_per_worker
        self.num_gpus = num_gpus
        self.terminate = False
        self.eval_in_queue = eval_in_queue
        self.eval_out_queue = eval_out_queue
        self.remote = remote
        self.kdt = kdt  # centroidal voronoi tessalations

        self.process = TorchProcess(target=self._run, daemon=True)
        self.process.start()

    def _run(self):
        if self.num_gpus:
            nvmlInit()  # for tracking available gpu resources

        # start the simulations
        envs = [self.env_fn_wrappers.x[i]() for i in range(len(self.env_fn_wrappers.x))]

        # begin the process loop
        while not self.terminate:
            evolved_actor_keys, eval_id = self.eval_in_queue.get(block=True, timeout=1e9)  # TODO: make sure this interface is implemented correctly
            actors = self.all_evolved_actors[evolved_actor_keys]
            assert len(envs) % len(actors) == 0, 'Number of envs should be a multiple of the number of policies '
            gpu_id = get_least_busy_gpu(self.num_gpus) if self.num_gpus else None
            device = torch.device(f'cuda:{gpu_id}' if (torch.cuda.is_available() and gpu_id is not None) else 'cpu')
            batch_actors = BatchMLP(actors, device)
            num_actors = len(actors)
            for env in envs:
                env.seed(int((self.master_seed * 100) * eval_id))

            obs = torch.from_numpy(np.array([env.reset() for env in envs])).reshape(num_actors, -1).to(actors[0].device)
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
                        rews[idx] += rew
                        obs_arr.append(obs)
                        dones[idx] = done
                        infos[idx] = info
                    obs = torch.from_numpy(np.array(obs_arr)).reshape(num_actors, -1).to(actors[0].device)
            ep_lengths = [env.ep_length for env in envs]
            bds = [info['desc'] for info in infos]  # list of behavioral descriptors
            res = [[rew, ep_len, bd] for rew, ep_len, bd in zip(rews, ep_lengths, bds)]

            self._map_agents(actors, bds, rews)

    def _terminate(self):
        self.terminate = True

    def _map_agents(self, actors, descs, rews):
        '''
        Map the evaluated agents using their behavior descriptors
        :param behav_descs: behavior descriptors of a batch of evaluated agents
        :param rews: fitness scores of a batch of evaluated agents
        '''
        for actor, desc, rew in zip(actors, descs, rews):
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
                    self.all_actors[map_agent_id] = agent
            else:
                # need to find a new, unused agent id since this agent maps to a new cell
                agent_id = self._find_available_agent_id()
                self.elites_map[n] = (agent_id, agent.fitness)
                self.all_actors[agent_id] = agent

    def _find_available_agent_id(self):
        '''
        If an agent maps to a new cell in the elites map, need to give it a new, unused id from the pool of agents
        :returns an agent id that's not being used by some other agent in the elites map
        '''
        # TODO: implement
        return -1



