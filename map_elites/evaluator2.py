import torch
import torch.nn as nn
import numpy as np
import time

from itertools import count
from utils.vectorized import BatchMLP
from utils.logger import log
from map_elites import common as cm
from collections import deque

# flags for the archive
UNUSED = 0
MAPPED = 1

class Individual(object):
    _ids = count(0)

    def __init__(self, genotype, parent_1_id, parent_2_id, genotype_type, genotype_novel, genotype_delta_f, phenotype, fitness, centroid=None):
        """
        A single agent
        param genotype:  The parameters that produced the behavior. I.e. neural network, etc.
        param phenotype: the resultant behavior i.e. the behavioral descriptor
        param fitness: the fitness of the model. In the case of a neural network, this is the total accumulated rewards
        param centroid: the closest CVT generator (a behavior) to the behavior that this individual exhibits
        """
        self.genotype = genotype
        self.genotype_id = self.get_next_id()
        self.parent_1_id = parent_1_id
        self.parent_2_id = parent_2_id
        self.genotype_type = genotype_type
        self.genotype_novel = genotype_novel
        self.genotype_delta_f = genotype_delta_f
        self.phenotype = phenotype
        self.fitness = fitness
        self.centroid = centroid
        self.novelty = None

    def get_next_id(self):
        next_id = next(self._ids)
        Individual.current_id = next_id  # TODO: not sure what this is for
        return next_id


class Evaluator(object):
    def __init__(self,
                 env,
                 all_actors,
                 eval_cache,
                 elites_map,
                 batch_size,
                 seed,
                 num_gpus,
                 kdt,
                 eval_in_queue):
        '''
        A class for batch evaluations of mutated policies
        '''

        self.env = env
        self.all_actors = all_actors
        self.eval_cache = eval_cache
        self.elites_map = elites_map
        self.batch_size = batch_size
        self.seed = seed
        self.num_gpus = num_gpus
        self.kdt = kdt
        self.eval_in_queue = eval_in_queue
        self.eval_id = 0

        # logging variables
        self.avg_stats_intervals = (2, 12, 60)  # 10 seconds, 1 minute, 5 minutes
        self._report_interval = 5.0  # report every 5 seconds
        self._last_report = 0.0
        self.eval_stats = deque([], maxlen=max(self.avg_stats_intervals))


    def evaluate_batch(self):
        # TODO, replace this with signal_slot model
        mutated_actor_keys = self.eval_in_queue.get(block=True, timeout=1e9)
        start_time = time.time()
        actors = self.eval_cache[mutated_actor_keys]

        device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_actors = BatchMLP(actors, device)
        num_actors = len(actors)
        self.env.seed(int((self.seed * 100) * self.eval_id))
        self.eval_id += 1
        obs = self.env.reset()['obs']  # isaac gym returns dict that contains obs

        rews = [0 for _ in range(num_actors)]
        dones = [False for _ in range(num_actors)]
        infos = [None for _ in range(num_actors)]
        # get a batch of trajectories and rewards
        while not all(dones):
            obs_arr = []
            with torch.no_grad():
                acts = batch_actors(obs)
                for idx, (act, env) in enumerate(zip(acts, self.envs)):
                    obs, rew, done, info = env.step(act)
                    rews[idx] += rew * (1 - dones[idx])
                    obs_arr.append(obs)
                    dones[idx] = done
                    infos[idx] = info
                obs = torch.from_numpy(np.array(obs_arr)).reshape(num_actors, -1).to(device)
        ep_lengths = [env.ep_length for env in self.envs]
        frames = sum(ep_lengths)
        bds = [info['desc'] for info in infos]  # list of behavioral descriptors

        runtime = time.time() - start_time
        metadata, runtime, frames, evals = self._map_agents(actors, mutated_actor_keys, bds, rews, runtime, frames)
        log.debug(f'Processed batch of {len(actors)} agents in {round(runtime, 1)} seconds')
        return metadata, runtime, frames, evals

    def _map_agents(self, actors, actor_keys, descs, rews, runtime, frames):
        '''
        Map the evaluated agents using their behavior descriptors. Send the metadata back to the main process for logging
        :param behav_descs: behavior descriptors of a batch of evaluated agents
        :param rews: fitness scores of a batch of evaluated agents
        '''
        metadata = []
        evals = len(actors)
        for actor, actor_key, desc, rew in zip(actors, actor_keys, descs, rews):
            added = False
            agent = Individual(genotype=actor_key, parent_1_id=actor.parent_1_id, parent_2_id=actor.parent_2_id,
                               genotype_type=actor.type, genotype_novel=actor.novel, genotype_delta_f=actor.delta_f,
                               phenotype=desc, fitness=rew)
            actor.id = agent.get_next_id()
            niche_index = self.kdt.query([desc], k=1)[1][0][
                0]  # get the closest voronoi cell to the behavior descriptor
            niche = self.kdt.data[niche_index]
            n = cm.make_hashable(niche)
            agent.centroid = n
            if n in self.elites_map:
                # natural selection
                map_agent_id, fitness = self.elites_map[
                    n]  # TODO: make sure this interface is correctly implemented
                if agent.fitness > fitness:
                    self.elites_map[n] = (map_agent_id, agent.fitness)
                    agent.genotype = map_agent_id
                    # override the existing agent in the actors pool @ map_agent_id. This species goes extinct b/c a more fit one was found
                    self.all_actors[map_agent_id] = (actor, MAPPED)
                    added = True
            else:
                # need to find a new, unused agent id since this agent maps to a new cell
                agent_id = self._find_available_agent_id()
                self.elites_map[n] = (agent_id, agent.fitness)
                agent.genotype = agent_id
                self.all_actors[agent_id] = (actor, MAPPED)
                added = True

            if added:
                md = (
                agent.genotype_id, agent.fitness, str(agent.phenotype).strip("[]"), str(agent.centroid).strip("()"),
                agent.parent_1_id, agent.parent_2_id, agent.genotype_type, agent.genotype_novel,
                agent.genotype_delta_f)
                metadata.append(agent)

        return metadata, runtime, frames, evals

    def _find_available_agent_id(self):
        '''
        If an agent maps to a new cell in the elites map, need to give it a new, unused id from the pool of agents
        :returns an agent id that's not being used by some other agent in the elites map
        '''
        return next(i for i in range(len(self.all_actors)) if self.all_actors[i][1] == UNUSED)

    def close_envs(self):
        for env in self.envs:
            env.close()


    @property
    def report_interval(self):
        return self._report_interval

    @property
    def last_report(self):
        return self._last_report

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