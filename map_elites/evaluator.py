import torch
import torch.nn as nn
import numpy as np
import time
import gc

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
MUTATED = 2
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


class Evaluator():
    def __init__(self,
                 cfg,
                 all_actors,
                 elites_map,
                 eval_in_queue,
                 batch_size,
                 seed,
                 num_gpus,
                 kdt,
                 gpu_id):
        '''
        A class for batch evaluations of mutated policies
        '''
        self.cfg = cfg
        self.vec_env = None
        self.high, self.low = None, None
        self.action_space = None
        self.all_actors = all_actors
        self.elites_map = elites_map
        self.eval_in_queue = eval_in_queue
        self.batch_size = batch_size
        self.seed = seed
        self.num_gpus = num_gpus
        self.kdt = kdt
        self.eval_id = 0
        self.gpu_id = gpu_id
        self.sim_device = f'cuda:{self.gpu_id}'

    def init_env(self):
        self.vec_env = make_gym_env(cfg=self.cfg, graphics_device_id=self.gpu_id, sim_device=self.sim_device)
        self.high = torch.tensor(self.vec_env.env.action_space.high).to(self.sim_device)
        self.low = torch.tensor(self.vec_env.env.action_space.low).to(self.sim_device)
        self.action_space = self.vec_env.env.action_space

    def resize_env(self, num_envs):
        self.cfg.num_agents = num_envs
        self.vec_env.env.destroy()
        del self.vec_env
        gc.collect()
        self.vec_env = make_gym_env(cfg=self.cfg, graphics_device_id=self.gpu_id, sim_device=self.sim_device)

    def on_evaluate(self, var_id, init_mode):
        '''
        Evaluate a new batch of mutated policies
        :param var_id: Variation Worker that mutated the actors
        :param mutated_actor_keys: locations in the eval cache that hold the mutated policies
        '''
        mutated_actor_keys = self.eval_in_queue.get(block=True, timeout=1e6)
        self.evaluate_batch(mutated_actor_keys, init_mode)

    def evaluate_batch(self, actors, mutated_actor_keys):
        start_time = time.time()
        device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
        batch_actors = BatchMLP(actors, device)
        num_actors = len(actors)

        # TODO: Hack?
        if self.vec_env.env.num_envs // self.cfg.num_envs_per_policy != num_actors:
            log.warn(f'Early return from Evaluator\'s evaluate_batch() method because the '
                     f'vec_env object was not resized in time for the new batch of mutated actors. Num envs: {self.vec_env.env.num_envs}, '
                     f' Num actors: {num_actors}, envs per policy: {self.cfg.num_envs_per_policy}. Releasing keys...')
            return mutated_actor_keys

        self.vec_env.seed(int((self.seed * 100) * self.eval_id))
        self.eval_id += 1
        obs = self.vec_env.reset()['obs']  # isaac gym returns dict that contains obs

        rews = torch.zeros((self.vec_env.env.num_environments,)).to(device)
        cumulative_rews = [[] for _ in range(self.vec_env.env.num_environments)]
        dones = torch.zeros((self.vec_env.env.num_environments,))
        traj_len = 1000  # trajectory length for ant env, TODO: make this generic
        # get a batch of trajectories and rewards
        while not all(dones):
            with torch.no_grad():
                logits = batch_actors(obs).view(-1)
                acts = batch_actors.get_actions(self.action_space, logits, batch_actors.action_log_std.exp()).view(num_actors, -1) \
                    if self.cfg.continuous_actions_sample else logits.view(self.vec_env.env.num_environments, -1)
                acts = torch.clip(acts, self.low, self.high)  # isaacgym action spaces are all [-1, 1]
                obs, rew, dones, info = self.vec_env.step(acts)
                rews += rew
                for i, done in enumerate(dones):
                    if done:
                        # the vec_env will auto reset the env at this index, so we reset our reward counter
                        cumulative_rews[i].append(rews[i].cpu().numpy())
                        rews[i] = 0

                ep_lengths = info['ep_lengths']
                if ep_lengths.__contains__(traj_len):  # only collect fixed number of transitions
                    dones = torch.ones_like(dones)

        runtime = time.time() - start_time
        log.debug(f'Processed batch of {len(actors)} agents in {round(runtime, 1)} seconds')


        ep_lengths = info['ep_lengths']
        avg_ep_length = torch.mean(ep_lengths)
        frames = sum(ep_lengths).cpu().numpy()
        bds = info['desc'].cpu().numpy().reshape(num_actors, self.cfg.num_envs_per_policy, -1)  # behavior descriptors
        bds = np.mean(bds, axis=1)
        mean_rewards = np.vstack([sum(cumulative_rews[i]) / len(cumulative_rews[i]) for i in range(len(cumulative_rews))]).reshape(num_actors, self.cfg.num_envs_per_policy)
        mean_rewards = np.mean(mean_rewards, axis=1)

        agents = []
        mutated_actor_keys_repeated = np.repeat(mutated_actor_keys, repeats=self.cfg.mutations_per_policy)  # repeat keys M times b/c actors parents were mutated M times
        for actor, actor_key, desc, rew in zip(actors, mutated_actor_keys_repeated, bds, mean_rewards):
            agent = Individual(genotype=actor_key, parent_1_id=actor.parent_1_id, parent_2_id=actor.parent_2_id,
                               genotype_type=actor.type, genotype_novel=actor.novel, genotype_delta_f=actor.delta_f,
                               phenotype=desc, fitness=rew)
            agents.append(agent)
        return agents, mutated_actor_keys, frames, runtime, avg_ep_length

    def close_envs(self):
        self.vec_env.close()

    def on_stop(self):
        self.close_envs()
        log.debug('Done!')