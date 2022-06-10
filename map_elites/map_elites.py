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
from map_elites.evaluator import Evaluator
from map_elites.trainer import Trainer
from utils.vectorized import BatchMLP, combine
from torch.multiprocessing import Process as TorchProcess, Pipe
from functools import partial

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
    # all_actors = np.array([(ant_model_factory(device, hidden_size=cfg.hidden_size), UNUSED) for _ in range(n_niches * cfg.mutations_per_policy)])

    # save all actors in one giant BatchMLP object to reduce the number of open file descriptors and shared memory usage (b/c of this bug: https://github.com/pytorch/pytorch/issues/78274)
    model_fn = partial(ant_model_factory)
    all_actors = []
    for _ in range(cfg.mutations_per_policy): # need to do it this way to reduce number of open file descriptors at any given time, also b/c of the bug from previous comment
        mlps = [ant_model_factory(device, hidden_size=cfg.hidden_size, share_memory=True) for _ in range(n_niches)]
        batch_mlps = BatchMLP(cfg, device, model_fn, np.array(mlps))
        all_actors.append(batch_mlps)
    all_actors = combine(all_actors)


    # keep track of Individuals()
    agent_archive = []

    # create the CVT
    cluster_centers = cm.cvt(n_niches, cfg['dim_map'], cfg['cvt_samples'], cfg['cvt_use_cache'])
    kdt = KDTree(cluster_centers, leaf_size=30, metric='euclidean')
    cm.__write_centroids(cluster_centers)

    # create the map of elites
    elites_map = manager.dict()

    # sync access to list of free keys b/w all processes
    free_policy_keys = manager.list(range(n_niches * cfg.mutations_per_policy))

    # shared queues to keep track of which policies are free to mutate/being evaluated/being mapped to archive
    eval_in_queue = Queue(max_size_bytes=int(1e5))

    runner = Runner(cfg, agent_archive, elites_map, all_actors, actors_file, filename)

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
                              gpu_id=GPU_ID)

        variation_op = VariationOperator(cfg,
                                         all_actors,
                                         elites_map,
                                         eval_in_queue,
                                         free_policy_keys,
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

        trainer = Trainer(cfg, all_actors, elites_map, kdt, mutator=variation_op, evaluator=evaluator, event_loop=trainer_loop_i.event_loop, object_id=f'trainer {i}')
        trainers.append(trainer)
        trainer_loops.append(trainer_loop_i)
        GPU_ID = (GPU_ID + 1) % cfg.num_gpus

    runner.init_loops(trainers)

    for trainer_loop in trainer_loops:
        trainer_loop.start()
    runner.run()
    for trainer_loop in trainer_loops:
        trainer_loop.join()
