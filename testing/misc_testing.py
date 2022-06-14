from envs.isaacgym.make_env import make_gym_env
import numpy as np
import time
import pandas as pd
from models.bipedal_walker_model import BipedalWalkerNN
from models.ant_model import AntNN, ant_model_factory
import torch
from torch.multiprocessing import Process, Value
from multiprocessing import shared_memory
import torch.multiprocessing as multiprocessing
from enjoy_bipedal_walker import enjoy
from faster_fifo import Queue
from functools import partial
from collections import deque
from utils.vectorized import BatchMLP
from attrdict import AttrDict
from envs.isaacgym.make_env import make_gym_env
import copy
import psutil
from modelsize import SizeEstimator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sigma = np.ones(1000)

# you can ignore this file. For testing only

# 448986,
# 703041 - hopping
# 663801 - fast drag
# 528046
# 385292 - walking
# 581804
# 701314 - walking

def job1(a, t=None):
    s1 = a[0]
    s2 = Species()
    # with torch.no_grad():
    #     for p1, p2 in zip(s1.parameters(), s2.parameters()):
    #         p1.copy_(p2)
    with torch.no_grad():
        s1.load_state_dict(s2.state_dict())

def job2(a, t=None):
    s = a[0]
    s.p1 += 1

def job3(arr, idx):
    arr = copy.deepcopy(arr)
    arr[idx][idx][idx] = 1


class Species(AntNN):
    def __init__(self):
        super().__init__()
        self.p1 = torch.tensor(-1).to(device).share_memory_()
        self.p2 = None
        self.t = torch.tensor(5).to(device)

    def summary(self):
        print(f'{self.p1=}, {self.p2=}, {s.t=}')
        for n, p in self.named_parameters():
            print(n, p)




if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)  # cuda only works with this method
    except RuntimeError:
        pass
    # filepath = './checkpoints/checkpoint_002790022/archive_CVT-MAP-ELITES_BipedalWalkerV3_seed_0_dim_map_2_2790022.dat'
    # df = pd.read_csv(filepath, sep=' ')
    # df = df.to_numpy()[:, :-1]
    # elites = np.where(df[:, 0] >= 240)
    # df_elites = df[elites]
    # df_elites_sorted = np.array(sorted(df_elites, key=lambda x: x[0], reverse=True))
    # print(df_elites_sorted[:, [0, 3, 4, 5]])
    # inds = list(range(df_elites_sorted.shape[0]))
    # np.random.shuffle(inds)
    # rand_policies = df_elites_sorted[inds][:,-1]
    # for policy_id in rand_policies[:10]:
    #     print(f'Running policy {int(policy_id)}')
    #     policy_path = f'checkpoints/checkpoint_002790022/policies/CVT-MAP-ELITES_BipedalWalkerV3_seed_0_dim_map_2_actor_{int(policy_id)}.pt'
    #     enjoy(policy_path, render=True)


    # t = torch.tensor(1).to(torch.device('cpu')).share_memory_()
    # s = Species().share_memory()
    # a = [s]
    # print('Before: ', a[0].summary())
    # p1 = Process(target=job1, args=(a, t))
    # p2 = Process(target=job2, args=(a, t))
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
    # print('After: ', a[0].summary())
    # print(t)
    #
    # arr = [[torch.tensor([0, 0, 0, 0, 0]).to(torch.device('cpu')).share_memory_() for _ in range(3)] for _ in range(5)]
    # p3 = Process(target=job3, args=(arr, 0))
    # p4 = Process(target=job3, args=(arr, 2))
    # p3.start()
    # p4.start()
    # p3.join()
    # p4.join()
    # print(arr)



    # num_agents = 1000
    # cfg = AttrDict({'headless': True, 'num_agents': num_agents})
    # vec_env = make_gym_env(cfg)
    # mlps = []
    # device = torch.device('cuda:0')
    # for _ in range(num_agents):
    #     mlps.append(ant_model_factory(device))
    # create_start = time.time()
    # batch_mlp = BatchMLP(np.array(mlps), device)
    # create_time = time.time() - create_start
    # print(f'Took {create_time:.2f} seconds to make a batch mlp of {num_agents} agents')
    # single_mlp = ant_model_factory(device)
    #
    # data = torch.randn((num_agents, 60)).to(device)
    # batch_start = time.time()
    # for _ in range(1000):
    #     acts = batch_mlp(data)
    #     vec_env.step(acts)
    # batch_time = time.time() - batch_start
    #
    # single_start = time.time()
    # for _ in range(1000):
    #     acts = single_mlp(data)
    #     vec_env.step(acts)
    # single_time = time.time() - single_start
    #
    # print(f'Batch MLP took {batch_time:.2f} seconds, Single MLP took {single_time:.2f} seconds')
    #
    # backup_ids, backup_mlps = list(range(1000)), [ant_model_factory(device) for _ in range(1000)]
    # for j in range(len(backup_mlps)):
    #     for i in range(len(backup_mlps[j].layers)):
    #         if not isinstance(backup_mlps[j].layers[i], torch.nn.Linear):
    #             continue
    #         backup_mlps[j].layers[i].weight = torch.nn.Parameter(torch.zeros_like(backup_mlps[j].layers[i].weight))
    #
    # replace_start = time.time()
    # batch_mlp.replace_mlps(backup_ids, backup_mlps)
    # replace_time = time.time() - replace_start
    #
    # print(f'Took {replace_time} seconds to replace {len(backup_mlps)} MLPs')

    # n_niches = 1024
    # mutations_per_policy = 10
    # num_policies = int(n_niches * mutations_per_policy)
    #
    # t = torch.ones((10240, 41872)).to(torch.device('cpu')).share_memory_()
    # mlp = ant_model_factory(torch.device('cpu'), 128, share_memory=True)
    # se = SizeEstimator(mlp, input_size=(10, 60))
    # print(se.estimate_size())
    # print(f'{num_policies=}')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # time.sleep(3)
    # device = torch.device('cpu')
    # mlps = []
    # for _ in range(num_policies):
    #     mlp = ant_model_factory(device, 128, share_memory=True)
    #     mlps.append(mlp)
    #     torch.cuda.empty_cache()
    #     print(f'Num mlps: {len(mlps)}')
    #     print(f'RAM Memory % used: {psutil.virtual_memory()[2]}')
    #
    # print(f'{len(mlps)=}')


    num_mlps = 1000
    mlps = []
    for i in range(num_mlps):
        mlp = ant_model_factory(device, hidden_size=128, share_memory=True)
        mlps.append(mlp)

    model_fn = partial(ant_model_factory)
    b = BatchMLP({'hidden_size': 128}, device, model_fn, np.array(mlps))
    new_mlps = []
    for i in range(2):
        new_mlp = ant_model_factory(device, hidden_size=128, share_memory=True)
        new_mlps.append(new_mlp)
    new_mlps[0].layers[0].weight.data = torch.zeros_like(new_mlps[0].layers[0].weight.data)
    b[42, 69] = new_mlps
    print(b)


