import sys
import argparse
import numpy as np
from envs.isaacgym.make_env import make_gym_env
from train_isaac import str2bool
from attrdict import AttrDict

import torch
from models.ant_model import ant_model_factory
from vectorized import BatchMLP
from utils.logger import log

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_path', type=str, help='path to the policy')
    parser.add_argument('--render', default=True, type=str2bool, help='Render the environment')
    parser.add_argument('--repeat', default=False, type=str2bool, help='Repeat the same policy in a loop until keyboard interrupt. Disable if you want to visualize multiple policies in one run')
    args = parser.parse_args()
    return args


def act_sense(cfg, env, actor, obs_low, obs_high, repeat):
    while True:
        obs = env.reset()['obs']
        rews = torch.zeros(cfg.num_agents).to(device)
        cumulative_rewards = [[] for _ in range(cfg.num_agents)]
        done = False
        try:
            for _ in range(1000):
                with torch.no_grad():
                    acts = actor(obs)
                    acts = torch.clip(acts, obs_low, obs_high)
                    obs, rew, dones, info = env.step(acts)
                    rews += rew
                    for i, done in enumerate(dones):
                        if done:
                            log.warning('RESET')
                            cumulative_rewards[i].append(rews[i].cpu().numpy())
                            rews[i] = 0

            for i in range(cfg.num_agents):
                if rews[i] != 0:  # for each agent, if sim ended at T=1000 without triggering the done flag, then there will be one hanging episode that wasn't stored in cumulative-rewards
                    cumulative_rewards[i].append(rews[i].cpu().numpy())

            bd = info['desc']
            print(f'BD: {bd.detach().cpu().numpy()}, rew: {rews.detach().cpu().numpy()}')
            mean_rewards = np.vstack([sum(cumulative_rewards[i]) / len(cumulative_rewards[i]) for i in range(len(cumulative_rewards))])
            print(f'{mean_rewards=}')
            if not repeat:
                break
        except KeyboardInterrupt:
            break


def enjoy(cfg, policy_path, render=True, repeat=False):
    cfg['num_agents'] = 1
    cfg['headless'] = not render
    cfg['mutations_per_policy'] = 1
    env = make_gym_env(cfg)
    obs_high = torch.tensor(env.env.action_space.high).to(device)
    obs_low = torch.tensor(env.env.action_space.low).to(device)

    actor = ant_model_factory(device, hidden_size=128, share_memory=False)
    actor.load(policy_path)
    actor.to(device)

    act_sense(cfg, env, actor, obs_low, obs_high, repeat)

    env.env.destroy()
    del env


def enjoy_many(cfg, policy_paths=None, render=True, repeat=False):
    cfg['num_agents'] = len(policy_paths)
    cfg['headless'] = not render
    cfg['mutations_per_policy'] = 1
    env = make_gym_env(cfg)
    obs_high = torch.tensor(env.env.action_space.high).to(device)
    obs_low = torch.tensor(env.env.action_space.low).to(device)

    actors = []
    for path in policy_paths:
        actor = ant_model_factory(device, hidden_size=128, share_memory=False)
        actor.load(path)
        actor.to(device)
        actors.append(actor)

    actors = BatchMLP(device, np.array(actors))
    act_sense(cfg, env, actors, obs_low, obs_high, repeat)  # main simulation loop

    env.env.destroy()
    del env


if __name__ == '__main__':
    args = parse_args()
    cfg = AttrDict(vars(args))
    sys.exit(enjoy(cfg, cfg.policy_path, cfg.render, cfg.repeat))
