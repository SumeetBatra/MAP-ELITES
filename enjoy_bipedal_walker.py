import sys
import gym
import torch
import argparse
from wrappers.BDWrapper import BDWrapper
from models.bipedal_walker_model import BipedalWalkerNN
from train_bipedal_walker import str2bool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_path', type=str, help='path to the policy')
    parser.add_argument('--render', default=True, type=str2bool, help='Render the environment')
    parser.add_argument('--repeat', default=False, type=str2bool, help='Repeat the same policy in a loop until keyboard interrupt. Disable if you want to visualize multiple policies in one run')
    args = parser.parse_args()
    return args

def make_env(env_name='BipedalWalker-v3'):
    assert env_name in ['BipedalWalker-v3'], 'currently only BipedalWalker-v3 is supported'
    env = gym.make(env_name)
    env = BDWrapper(env)
    return env


def enjoy(policy_path, render=True, repeat=False):
    actor = BipedalWalkerNN(device, hidden_size=128)
    actor.load(policy_path)

    env = make_env()
    while repeat:
        obs = env.reset()
        done = False
        try:
            while not done:
                obs = torch.from_numpy(obs)
                if render: env.render()
                with torch.no_grad():
                    action = actor(obs).detach().cpu().numpy()
                obs, rew, done, info = env.step(action)
        except KeyboardInterrupt:
            break
    env.close()
    return 1


if __name__ == '__main__':
    args = parse_args()
    sys.exit(enjoy(args.policy_path, args.render, args.repeat))