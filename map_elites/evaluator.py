import torch
import torch.nn as nn
import numpy as np
import time

from torch.multiprocessing import Value
from itertools import count
from utils.vectorized import BatchMLP
from utils.logger import log
from utils.signal_slot import EventLoopObject, signal, EventLoopProcess
from map_elites import common as cm
from collections import deque
from envs.isaacgym.make_env import make_gym_env

# flags for the archive
UNUSED = 0
MAPPED = 1
TO_EVALUATE = 2
EVALUATED = 3

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


class Evaluator(EventLoopObject):
    def __init__(self,
                 cfg,
                 all_actors,
                 eval_cache,
                 elites_map,
                 eval_in_queue,
                 batch_size,
                 seed,
                 num_gpus,
                 kdt,
                 event_loop,
                 object_id,
                 gpu_id):
        '''
        A class for batch evaluations of mutated policies
        '''
        super().__init__(event_loop, object_id)
        self.cfg = cfg
        self.vec_env = None
        self.all_actors = all_actors
        self.eval_cache = eval_cache
        self.elites_map = elites_map
        self.eval_in_queue = eval_in_queue
        self.batch_size = batch_size
        self.seed = seed
        self.num_gpus = num_gpus
        self.kdt = kdt
        self.eval_id = 0
        self.gpu_id = gpu_id
        self.sim_device = f'cuda:{self.gpu_id}'

    @signal
    def stop(self): pass

    @signal
    def eval_results(self): pass

    @signal
    def request_new_batch(self): pass

    @signal
    def request_from_init_map(self): pass  # send this signal until init_map() has produced enough policies

    @signal
    def init_elites_map(self): pass

    def init_env(self):
        self.vec_env = make_gym_env(cfg=self.cfg, graphics_device_id=self.gpu_id, sim_device=self.sim_device)
        self.init_elites_map.emit()

    def on_evaluate(self, var_id, init_mode):
        '''
        Evaluate a new batch of mutated policies
        :param var_id: Variation Worker that mutated the actors
        :param mutated_actor_keys: locations in the eval cache that hold the mutated policies
        '''
        mutated_actor_keys = self.eval_in_queue.get()
        self.evaluate_batch(mutated_actor_keys, init_mode)

    def evaluate_batch(self, mutated_actor_keys, init_mode=False):
        start_time = time.time()
        actors = self.eval_cache[mutated_actor_keys]
        device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
        batch_actors = BatchMLP(actors, device)
        num_actors = len(actors)

        self.vec_env.seed(int((self.seed * 100) * self.eval_id))
        self.eval_id += 1
        obs = self.vec_env.reset()['obs']  # isaac gym returns dict that contains obs

        rews = torch.zeros((num_actors,)).to(device)
        dones = torch.zeros((num_actors,))
        # get a batch of trajectories and rewards
        while not all(dones):
            with torch.no_grad():
                acts = batch_actors(obs)
                obs, rew, dones, info = self.vec_env.step(acts)
                rews += rew

        runtime = time.time() - start_time
        log.debug(f'Processed batch of {len(actors)} agents in {round(runtime, 1)} seconds')

        # if init_mode: # if init_map() is running, we don't need to do this
        #     # queue up a new batch of agents to evolve while the evaluator finishes processing this batch
        #     self.request_from_init_map.emit()

        ep_lengths = info['ep_lengths']
        frames = sum(ep_lengths).cpu().numpy()
        bds = info['desc'].cpu().numpy()  # behavior descriptors
        rews = rews.cpu().numpy()

        agents = []
        for actor, actor_key, desc, rew in zip(actors, mutated_actor_keys, bds, rews):
            agent = Individual(genotype=actor_key, parent_1_id=actor.parent_1_id, parent_2_id=actor.parent_2_id,
                               genotype_type=actor.type, genotype_novel=actor.novel, genotype_delta_f=actor.delta_f,
                               phenotype=desc, fitness=rew)
            agents.append(agent)
        self.eval_results.emit(self.object_id, agents, mutated_actor_keys, frames)

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
            niche_index = self.kdt.query([desc], k=1)[1][0][0]  # get the closest voronoi cell to the behavior descriptor
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
        self.vec_env.close()

    def on_stop(self, oid):
        self.close_envs()
        self.stop.emit(self.object_id)
        if isinstance(self.event_loop.process, EventLoopProcess):
            self.event_loop.stop()
        self.detach()
        log.debug('Done!')