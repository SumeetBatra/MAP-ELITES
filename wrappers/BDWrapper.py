import gym
import numpy as np


class BDWrapper(gym.Wrapper):
    def __init__(self, env, behavior_dim=2):
        """
        This wrapper gets the behavioral descriptor (the phenotype, the trajectory, etc.) that is used as a
        hashable into the map elites archive
        param behavior_dim: number of dimensions of the behavior. Ex. for bipedal walker, we care about the % time
        each leg is in contact with the ground, so the behavior dim is 2
        """
        super().__init__(env)
        self.ep_length = 0
        self.behavior_dim = behavior_dim
        self.leg1_idx, self.leg2_idx = 1, 3  # indices of the 2 actual legs in BipedalWalker
        self.behavior_desc = np.array([0.0 for _ in range(behavior_dim)])  # num_legs dim behavior descriptor = % time each foot is in contact with the ground
        self.behavior_desc_acc = np.array([0.0 for _ in range(behavior_dim)])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)  # use standard reward func: move forward without falling and minimizing energy consumption
        self.behavior_desc_acc += np.array([int(self.env.legs[self.leg1_idx].ground_contact), int(self.env.legs[self.leg2_idx].ground_contact)])
        self.ep_length += 1
        if done:
            self.behavior_desc = self.behavior_desc_acc / self.ep_length
            info['desc'] = self.behavior_desc
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        self.ep_length = 0
        self.behavior_desc = np.array([0.0 for _ in range(self.behavior_dim)])
        self.behavior_desc_acc = np.array([0.0 for _ in range(self.behavior_dim)])
        return obs
