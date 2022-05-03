import torch
import torch.nn as nn
import gym

from abc import ABC, abstractmethod
from torch.distributions import MultivariateNormal, Categorical


class Policy(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self.layers: nn.Sequential

    @abstractmethod
    def forward(self, obs):
        pass

    def get_actions(self, raw_logits, stddev=None):
        dist = self.get_action_distribution(raw_logits, stddev)
        return dist.sample()

    @staticmethod
    def get_action_distribution(action_space, raw_logits, scale=None):
        if isinstance(action_space, gym.spaces.Discrete):
            return Categorical(logits=raw_logits)
        if isinstance(action_space, gym.spaces.Box):
            assert scale is not None, "Must pass in the stddev vector!"
            cov_mat = torch.eye(len(scale)) * scale
            return MultivariateNormal(loc=raw_logits, covariance_matrix=cov_mat)

