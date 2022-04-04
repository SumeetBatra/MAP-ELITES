import torch
import numpy as np
import time
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import pandas as pd
from models.bipedal_walker_model import BipedalWalkerNN
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

def job1(x, dq):
    dq.append(x)



if __name__ == '__main__':
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
    lock = mp.Lock()
    x = 1
    dq = deque([])
    p1 = Process(target=job1, args=(x, dq))
    p2 = Process(target=job1, args=(x+1, dq))
    print(dq)
    p1.start()
    p2.start()