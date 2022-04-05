import torch.multiprocessing as multiprocessing
import os.path

import gym
import sys
import argparse
from functools import partial

import torch
import numpy as np

from multiprocessing import Pipe
from faster_fifo import Queue
from utils.vectorized import ParallelEnv
from utils.logger import log, config_wandb
from wrappers.BDWrapper import BDWrapper
from map_elites.cvt import compute_ht
from pynvml import *


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
    parser.add_argument('--num_workers', type=int, default=-1, help='# of cores to use. -1 means use all cores')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--n_niches', type=int, default=1296, help='number of niches/cells of behavior')
    parser.add_argument('--cvt_samples', type=int, default=25000, help='# of samples for computing cvt clusters. Larger value --> higher quality CVT')
    parser.add_argument('--batch_size', type=int, default=100, help='batch evaluations')
    parser.add_argument('--random_init', type=int, default=500, help='Number of random evaluations to initialize G')
    parser.add_argument('--random_init_batch', type=int, default=100, help='batch for random initialization')
    parser.add_argument('--cvt_use_cache', type=str2bool, default=True, help='do we cache results of CVT and reuse?')
    parser.add_argument('--max_evals', type=int, default=1e6, help='Total number of evaluations to perform')
    parser.add_argument('--save_path', default='./results', type=str, help='path where to save results')
    parser.add_argument('--dim_map', default=2, type=int, help='Dimensionality of the behavior space. Default is 2 for bipedal walker (obviously)')
    parser.add_argument('--save_period', default=500, type=int, help='How many evaluations b/w saving archives')
    parser.add_argument('--keep_checkpoints', default=2, type=int, help='Number of checkpoints of the elites to keep during training')
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str, help='Where to save the checkpoints')
    parser.add_argument('--cp_save_period', default=500, type=int, help='How many evaluations b/w saving checkpoints')
    parser.add_argument('--use_wandb', default=True, type=str2bool, help='log results to weights and biases')

    # args for parallelization
    parser.add_argument('--num_gpus', default=1, type=int, help='Number of gpus available on your system')
    parser.add_argument('--eval_workers_per_gpu', default=5, type=int, help='How many policy evaluators to run in parallel per gpu (different from number of policies to create a BatchMLP)')
    parser.add_argument('--actors_batch_size', default=10, type=int, help='Number of policies used to create a BatchMLP')
    parser.add_argument('--num_variation_workers', default=-1, type=int, help='Number of parallel processes performing crossover/mutation')

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


    #TODO: remove "parallel" parameter from compute() method. Should be based on --num_workers

    args = parser.parse_args()
    return args


def make_env(env_name='BipedalWalker-v3'):
    assert env_name in ['BipedalWalker-v3'], 'currently only BipedalWalker-v3 is supported'
    env = gym.make(env_name)
    env = BDWrapper(env)
    return env


def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)  # cuda only works with this method
    except RuntimeError:
        log.error("Cannot set mp start method to 'spawn'")

    # improve gpu memory usage
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    args = parse_args()
    cfg = vars(args)

    num_gpus = cfg['num_gpus']
    if num_gpus:
        nvmlInit()  # for tracking gpu resources

    # log hyperparams to wandb
    if cfg['use_wandb']:
        config_wandb(batch_size=cfg['batch_size'], max_evals=cfg['max_evals'])


    # set up factory function to launch parallel environments
    assert int(cfg['proportion_evo'] * cfg['eval_batch_size']) % cfg['actors_batch_size'] == 0 and \
           cfg['random_init_batch'] % cfg['actors_batch_size'] == 0, 'number of policies to evaluate during the init/eval phase must be a multiple of actors_batch_size'

    if num_gpus >= 1:
        env_fns = [[[partial(make_env) for _ in range(cfg['actors_batch_size'])] for _ in range(cfg['eval_workers_per_gpu'])] for _ in range(num_gpus)]
    else:
        env_fns = [[partial(make_env) for _ in range(cfg['actors_batch_size'])] for _ in range(cfg['eval_workers_per_gpu'])]

    # make folders
    if not os.path.exists(cfg['save_path']):
        os.mkdir(cfg['save_path'])
    if not os.path.exists(cfg['checkpoint_dir']):
        os.mkdir(cfg['checkpoint_dir'])

    log.debug(f'############## PARAMETERS #########################')
    for key, val in cfg.items():
        log.debug(f'{key}: {val}')
    log.debug('#' * 50)

    filename = f'CVT-MAP-ELITES_BipedalWalkerV3_seed_{cfg["seed"]}_dim_map_{cfg["dim_map"]}'
    file_save_path = os.path.join(cfg['save_path'], filename)
    actors_file = open(file_save_path, 'w')

    # set seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # get process num_workers input
    num_cores = multiprocessing.cpu_count()
    num_workers = cfg['num_workers']
    if num_workers == -1: num_workers = num_cores
    assert num_workers <= num_cores, '--num_workers must be less than or equal to the number of cores on your machine. Multiple workers per cpu are currently not supported'
    cfg['num_workers'] = num_workers  # for printing the cfg vars

    # same thing with variation workers
    num_var_workers = cfg['num_variation_workers']
    if num_var_workers == -1:
        cfg['num_variation_workers'] = num_cores
        num_var_workers = num_cores

    compute_ht(cfg,
               env_fns,
               num_var_workers,
               actors_file,
               filename,
               cfg['save_path'],
               n_niches=cfg['n_niches'],
               max_evals=cfg['max_evals'])

    return 1


if __name__ == '__main__':
    sys.exit(main())