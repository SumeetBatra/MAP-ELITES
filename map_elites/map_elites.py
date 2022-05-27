#! /usr/bin/env python

import copy
import math
import numpy as np

# from scipy.spatial import cKDTree : TODO -- faster?
import wandb
import time
import os
import shutil
import glob
import torch
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Value
from sklearn.neighbors import KDTree
from models.ant_model import ant_model_factory
from models.bipedal_walker_model import bpwalker_model_factory
from faster_fifo import Queue
from map_elites.variation import VariationOperator
from map_elites.evaluator import Evaluator, UNUSED, MAPPED
from map_elites.trainer import Trainer
from utils.vectorized import BatchMLP
from torch.multiprocessing import Process as TorchProcess, Pipe

from map_elites import common as cm
from utils.logger import log, config_wandb
from utils.signal_slot import EventLoopObject, EventLoopProcess, EventLoop
from utils.utils import *
from runner.runner import Runner

torch.multiprocessing.set_sharing_strategy('file_system')


EVAL_CACHE_SIZE = 500
GPU_ID = 0


def __add_to_archive(s, centroid, archive, kdt):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche)
    s.centroid = n
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s
        return 1


def compute_gpu(cfg, actors_file, filename, n_niches=1000):
    # for shared objects
    manager = multiprocessing.Manager()

    device = torch.device("cpu")  # batches of agents will be put on the least busiest gpu if cuda available during evolution/evaluation
    # initialize all actors
    # since tensors are using shared memory, changes to tensors in one process will be reflected across all processes
    all_actors = np.array([(ant_model_factory(device, hidden_size=cfg.hidden_size), UNUSED) for _ in range(n_niches * cfg.mutations_per_policy)])

    # # variation workers will put actors that need to be evaluated in here
    # eval_cache = []
    # for _ in range(n_niches):
    #     # eval cache will store M-batches of mlps since variation worker mutates M times per policy
    #     mlps = [ant_model_factory(device, hidden_size=cfg.hidden_size) for _ in range(cfg.mutations_per_policy)]
    #     eval_cache.append(mlps)
    # eval_cache = np.array(eval_cache)

    # keep track of Individuals()
    agent_archive = []

    # create the CVT
    cluster_centers = cm.cvt(n_niches, cfg['dim_map'], cfg['cvt_samples'], cfg['cvt_use_cache'])
    kdt = KDTree(cluster_centers, leaf_size=30, metric='euclidean')
    cm.__write_centroids(cluster_centers)

    # create the map of elites
    elites_map = manager.dict()

    # shared queues to keep track of which policies are free to mutate/being evaluated/being mapped to archive
    free_queue, eval_in_queue, map_queue = Queue(), Queue(), Queue()
    free_policy_keys = set(list(range(n_niches)))
    # init the free queue with keys for all policies
    for i in range(len(all_actors)):
        free_queue.put(i)

    runner = Runner(cfg, agent_archive, elites_map, all_actors, actors_file, filename)


    # do variation and evaluation in a subprocess
    # var_loop = EventLoopProcess('var_loop')
    # variation_op = VariationOperator(cfg,
    #                                  all_actors,
    #                                  elites_map,
    #                                  eval_cache,
    #                                  eval_in_queue,
    #                                  free_policy_keys,
    #                                  event_loop=var_loop.event_loop,
    #                                  object_id='variation worker',
    #                                  crossover_op=cfg.crossover_op,
    #                                  mutation_op=cfg.mutation_op,
    #                                  max_gene=cfg.max_genotype,
    #                                  min_gene=cfg.min_genotype,
    #                                  mutation_rate=cfg.mutation_rate,
    #                                  crossover_rate=cfg.crossover_rate,
    #                                  eta_m=cfg.eta_m,
    #                                  eta_c=cfg.eta_c,
    #                                  sigma=cfg.sigma,
    #                                  max_uniform=cfg.max_uniform,
    #                                  iso_sigma=cfg.iso_sigma,
    #                                  line_sigma=cfg.line_sigma)

    global GPU_ID
    trainers, trainer_loops = [], []
    for i in range(cfg.num_trainers):
        trainer_loop_i = EventLoopProcess(f'trainer loop {i}')
        evaluator = Evaluator(cfg,
                              all_actors,
                              elites_map,
                              eval_in_queue,
                              cfg.batch_size,
                              cfg.seed,
                              cfg.num_gpus,
                              kdt,
                              event_loop=trainer_loop_i.event_loop,
                              object_id=f'evaluator {i}',
                              gpu_id=GPU_ID)

        variation_op = VariationOperator(cfg,
                                         all_actors,
                                         elites_map,
                                         eval_in_queue,
                                         free_policy_keys,
                                         event_loop=trainer_loop_i.event_loop,
                                         object_id=f'mutator {i}',
                                         crossover_op=cfg.crossover_op,
                                         mutation_op=cfg.mutation_op,
                                         max_gene=cfg.max_genotype,
                                         min_gene=cfg.min_genotype,
                                         mutation_rate=cfg.mutation_rate,
                                         crossover_rate=cfg.crossover_rate,
                                         eta_m=cfg.eta_m,
                                         eta_c=cfg.eta_c,
                                         sigma=cfg.sigma,
                                         max_uniform=cfg.max_uniform,
                                         iso_sigma=cfg.iso_sigma,
                                         line_sigma=cfg.line_sigma)

        trainer = Trainer(all_actors, elites_map, kdt, mutator=variation_op, evaluator=evaluator, event_loop=trainer_loop_i.event_loop, object_id=f'trainer {i}')
        trainers.append(trainer)
        trainer_loops.append(trainer_loop_i)
        GPU_ID = (GPU_ID + 1) % cfg.num_gpus

    runner.init_loops(trainers)

    for trainer_loop in trainer_loops:
        trainer_loop.start()
    runner.run()
    for trainer_loop in trainer_loops:
        trainer_loop.join()
