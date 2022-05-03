import argparse
import sys
import numpy as np
import os

from utils.logger import log, config_wandb
from envs.isaacgym.make_env import make_gym_env
from attrdict import AttrDict
from map_elites.map_elites import compute_gpu

import torch.multiprocessing as multiprocessing
import torch

# train gpu accelerated isaac gym envs

# CLI args
def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ('true', ):
        return True
    elif isinstance(v, str) and v.lower() in ('false', ):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant', help='Which environment to train on')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--n_niches', type=int, default=1296, help='number of niches/cells of behavior')
    parser.add_argument('--cvt_samples', type=int, default=25000, help='# of samples for computing cvt clusters. Larger value --> higher quality CVT')
    parser.add_argument('--batch_size', type=int, default=100, help='batch evaluations')
    parser.add_argument('--random_init', type=int, default=500, help='Number of random evaluations to initialize G')
    parser.add_argument('--random_init_batch', type=int, default=100, help='batch for random initialization')
    parser.add_argument('--cvt_use_cache', type=str2bool, default=True, help='do we cache results of CVT and reuse?')
    parser.add_argument('--max_evals', type=int, default=1e6, help='Total number of evaluations to perform')
    parser.add_argument('--save_path', default='./results', type=str, help='path where to save results')
    parser.add_argument('--dim_map', default=4, type=int, help='Dimensionality of the behavior space. Default is 2 for bipedal walker (obviously)')
    parser.add_argument('--save_period', default=500, type=int, help='How many evaluations b/w saving archives')
    parser.add_argument('--keep_checkpoints', default=2, type=int, help='Number of checkpoints of the elites to keep during training')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str, help='Where to save the checkpoints')
    parser.add_argument('--cp_save_period_sec', default=300, type=int, help='How many seconds b/w saving checkpoints. Default saves every 5 min')
    parser.add_argument('--use_wandb', default=True, type=str2bool, help='log results to weights and biases')

    # args for parallelization
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus available on your system')
    parser.add_argument('--num_evaluators', default=1, type=int, help='Number of evaluators for parallel policy evaluation. Best to set this to the number of gpus available on your system')

    # args for cross over and mutation of agent params
    parser.add_argument('--mutation_op', default=None, type=str, choices=['polynomial_mutation', 'gaussian_mutation', 'uniform_mutation'], help='Type of mutation to perform. Leave as None to do no mutations')
    parser.add_argument('--crossover_op', default='iso_dd', type=str, choices=['sbx', 'iso_dd'], help='Type of crossover operation to perform')
    parser.add_argument("--min_genotype", default=False, type=float, help='Minimum value a gene in the genotype can take (if False no limit) (Set to False in GECCO paper)')
    parser.add_argument("--max_genotype", default=False, type=float, help='Maximum value a gene in the genotype can take (if False no limit) (Set to False in GECCO paper)')
    parser.add_argument('--mutation_rate', default=0.05, type=float, help='probability of gene to be mutated')
    parser.add_argument('--crossover_rate', default=0.75, type=float, help='probability of genotypes being crossed over')
    parser.add_argument("--eta_m", default=5.0, type=float, help='Parameter for polynomaial mutation (Not used in GECCO paper)')
    parser.add_argument("--eta_c", default=10.0, type=float, help='Parameter for Simulated Binary Crossover (Not used in GECCO paper)')
    parser.add_argument("--sigma", default=0.2, type=float, help='Sandard deviation for gaussian muatation (Not used in GECCO paper)')
    parser.add_argument("--iso_sigma", default=0.01, type=float, help='Gaussian parameter in iso_dd/directional variation (sigma_1)')
    parser.add_argument("--line_sigma", default=0.2, type=float, help='Line parameter in iso_dd/directional variation (sigma_2)')
    parser.add_argument("--max_uniform", default=0.1, type=float, help='Max mutation for uniform muatation (Not used in GECCO paper)')
    parser.add_argument('--eval_batch_size', default=100, type=int, help='Batch size for parallel evaluation of policies')
    parser.add_argument('--proportion_evo', default=0.5, type=float, help='Proportion of batch to use in GA variation (crossovers/mutations)')
    parser.add_argument('--mutations_per_policy', default=10, type=int, help='Number of times to mutate a single policy (policy is stored as a batch of mutated policies)')

    # args for isaac gym
    parser.add_argument('--num_agents', default=10, type=int, help='Number of parallel envs in vectorized env')
    parser.add_argument('--headless', default=True, type=str2bool, help='Choose whether or not to render the scene')

    args = parser.parse_args()
    return args


def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)  # cuda only works with this method
    except RuntimeError:
        log.error("Cannot set mp start method to 'spawn'")

    # improve gpu memory usage
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    args = parse_args()
    cfg = AttrDict(vars(args))

    if cfg.use_wandb:
        config_wandb(batch_size=cfg.batch_size, max_evals=cfg.max_evals)


    # make folders
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    if not os.path.exists(cfg.checkpoint_dir):
        os.mkdir(cfg.checkpoint_dir)

    log.debug(f'############## PARAMETERS #########################')
    for key, val in cfg.items():
        log.debug(f'{key}: {val}')
    log.debug('#' * 50)

    filename = f'CVT-MAP-ELITES_QDAnt_seed_{cfg.seed}_dim_map_{cfg.dim_map}'
    file_save_path = os.path.join(cfg.save_path, filename)
    actors_file = open(file_save_path, 'w')

    # set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    compute_gpu(cfg,
                actors_file,
                filename,
                cfg.n_niches)


if __name__ == '__main__':
    sys.exit(main())