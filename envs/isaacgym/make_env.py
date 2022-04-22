import gym
import yaml
import os
from envs.isaacgym.isaacgymenvs.tasks.ant import QDAnt
from attrdict import AttrDict
import torch

isaacgym_task_map = {
    'ant': QDAnt
}

class IsaacGymVecEnv(gym.Env):
    def __init__(self, isaacgym_env):
        self.env = isaacgym_env
        self.num_agents = isaacgym_env.num_envs

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.num_agents

    def render(self, mode='human'):
        self.env.render()  # ??


def make_gym_env(cfg=None, env_config=None, graphics_device_id=0, sim_device='cuda:0'):
    task_name = 'ant'

    cfg_dir = os.path.join(os.getcwd(), 'envs/isaacgym/cfg')
    cfg_file = os.path.join(cfg_dir, 'task', f'{task_name}.yaml')

    with open(cfg_file, 'r') as yaml_stream:
        task_cfg = yaml.safe_load(yaml_stream)

    task_cfg['env']['numEnvs'] = cfg.num_agents

    env = isaacgym_task_map[task_name](
        cfg=task_cfg,
        sim_device=sim_device,
        graphics_device_id=graphics_device_id,
        headless=cfg.headless
    )
    env = IsaacGymVecEnv(env)
    return env

if __name__ == '__main__':
    cfg = AttrDict({'headless': False, 'num_agents': 4})
    env = make_gym_env(cfg)
    num_agents = cfg.num_agents
    while True:
        acts = torch.rand(size=(num_agents, env.env.act_space.shape[0]))
        _, _, dones, _ = env.step(acts)
        if all(dones):
            env.reset()
        env.render()


