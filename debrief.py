import pandas as pd
import numpy as np
from enjoy_isaac import enjoy, enjoy_many, parse_args
from attrdict import AttrDict




if __name__ == '__main__':
    sim_many_actors = True
    num_actors_to_sim = 80
    filepath = './checkpoints/checkpoint_015213140/archive_CVT-MAP-ELITES_QDAnt_seed_0_dim_map_4_15213140.dat'
    df = pd.read_csv(filepath, sep=' ')
    df = df.to_numpy()[:, :-1]
    elites = np.where(df[:, 0] >= 1200)
    df_elites = df[elites]
    df_elites_sorted = np.array(sorted(df_elites, key=lambda x: x[0], reverse=True))
    print(df_elites_sorted[:, [0, 3, 4, 5]])
    inds = list(range(df_elites_sorted.shape[0]))
    np.random.shuffle(inds)
    rand_policies = df_elites_sorted[inds][:,-1]

    # one policy at a time
    if not sim_many_actors:
        for policy_id in rand_policies[:10]:
            print(f'Running policy {int(policy_id)}')
            policy_path = f'checkpoints/checkpoint_015213140/policies/CVT-MAP-ELITES_QDAnt_seed_0_dim_map_4_actor_{int(policy_id)}.pt'

            args = parse_args()
            cfg = AttrDict(vars(args))
            enjoy(cfg, policy_path, render=True)

    # sim many policies at the same time
    if sim_many_actors:
        policy_paths = []
        for policy_id in rand_policies[:num_actors_to_sim]:
            policy_path = f'checkpoints/checkpoint_015213140/policies/CVT-MAP-ELITES_QDAnt_seed_0_dim_map_4_actor_{int(policy_id)}.pt'
            policy_paths.append(policy_path)

        args = parse_args()
        cfg = AttrDict(vars(args))
        enjoy_many(cfg, policy_paths, render=True)