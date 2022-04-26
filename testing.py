from envs.isaacgym.make_env import make_gym_env
import numpy as np
import time
import pandas as pd
from models.bipedal_walker_model import BipedalWalkerNN
from models.ant_model import AntNN
import torch
from torch.multiprocessing import Process, Value
from multiprocessing import shared_memory
import torch.multiprocessing as multiprocessing
from enjoy_bipedal_walker import enjoy
from faster_fifo import Queue
from functools import partial
from collections import deque

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
    arr[idx] = 1


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
    t = torch.tensor(1).to(torch.device('cpu')).share_memory_()
    s = Species().share_memory()
    a = [s]
    print('Before: ', a[0].summary())
    p1 = Process(target=job1, args=(a, t))
    p2 = Process(target=job2, args=(a, t))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print('After: ', a[0].summary())
    print(t)

    arr = torch.tensor([0, 0, 0, 0, 0])
    p3 = Process(target=job3, args=(arr, 0))
    p4 = Process(target=job3, args=(arr, 2))
    p3.start()
    p4.start()
    p3.join()
    p4.join()
    print(arr)