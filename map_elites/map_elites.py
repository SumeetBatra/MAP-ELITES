#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
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
from sklearn.neighbors import KDTree
from models.ant_model import ant_model_factory
from faster_fifo import Queue
from map_elites.variation2 import VariationOperator
from map_elites.evaluator2 import Evaluator, Individual
from torch.multiprocessing import Process as TorchProcess, Pipe

from map_elites import common as cm
from utils.logger import log, config_wandb
from utils.utils import *

torch.multiprocessing.set_sharing_strategy('file_system')

# flags for the archive
UNUSED = 0
MAPPED = 1
TO_EVALUATE = 2
EVALUATED = 3

EVAL_CACHE_SIZE = 500


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


def compute_gpu(cfg, envs, actors_file, filename, save_path, n_niches=1000, max_evals=1e5):
    # for shared objects
    manager = multiprocessing.Manager()

    device = torch.device("cpu")  # batches of agents will be put on the least busiest gpu if cuda available during evolution/evaluation
    # initialize all actors
    # since tensors are using shared memory, changes to tensors in one process will be reflected across all processes
    all_actors = np.array([(ant_model_factory(device), UNUSED) for _ in range(n_niches)])
    # variation workers will put actors that need to be evaluated in here
    eval_cache = np.array([copy.deepcopy(model) for model, _ in all_actors])

    # keep track of Individuals()
    agent_archive = []

    # create the CVT
    cluster_centers = cm.cvt(n_niches, cfg['dim_map'], cfg['cvt_samples'], cfg['cvt_use_cache'])
    kdt = KDTree(cluster_centers, leaf_size=30, metric='euclidean')
    cm.__write_centroids(cluster_centers)

    # create the map of elites
    elites_map = manager.dict()

    # variables for logging
    n_evals = 0
    cp_evals = 0
    steps = 0

    eval_in_queue = Queue(max_size_bytes=int(1e6))

    variation_op = VariationOperator(cfg,
                                     all_actors,
                                     elites_map,
                                     eval_cache,
                                     eval_in_queue,
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

    evaluator = Evaluator(envs,
                          all_actors,
                          eval_cache,
                          elites_map,
                          cfg.batch_size,
                          cfg.seed,
                          cfg.num_gpus,
                          kdt,
                          eval_in_queue)

    # initialize the map of elites with a few initial random policies
    # TODO: replace this with signal-slot refactor
    rand_init_batch_size = cfg.random_init_batch
    log.debug(f'Initializing the map of elites with {cfg.random_init} policies')
    while len(elites_map) <= cfg.random_init:
        variation_op.init_map(rand_init_batch_size)
        evaluator.evaluate_batch()

    # main loop
    log.debug('Running main loop')
    try:
        while n_evals < max_evals:
            start_time = time.time()
            variation_op.evolve_batch()
            metadata, runtime, frames, evals = evaluator.evaluate_batch()

            n_evals += evals
            cp_evals += evals
            steps += frames
            fit_list = []
            if len(metadata) > 0:  # only write actors to file if new, better actors were added to the elites map
                for agent in metadata:
                    fit_list.append(agent.fitness)
                    agent_archive.append(agent)
                    actors_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(n_evals,
                                                                               agent.genotype_id,
                                                                               agent.fitness,
                                                                               agent.phenotype,
                                                                               agent.centroid,
                                                                               agent.parent_1_id,
                                                                               agent.parent_2_id,
                                                                               agent.genotype_type,
                                                                               agent.genotype_novel,
                                                                               agent.genotype_delta_f))
                    actors_file.flush()

                # write log
                log.info(f'n_evals: {n_evals}, mean fitness: {np.mean(fit_list)}, median fitness: {np.median(fit_list)}, \
                    5th percentile: {np.percentile(fit_list, 5)}, 95th percentile: {np.percentile(fit_list, 95)}')
                fit_list = np.array(fit_list)
                if cfg['use_wandb']:
                    wandb.log({
                        "evals": n_evals,
                        "mean fitness": np.mean(fit_list),
                        "median fitness": np.median(fit_list),
                        "5th percentile": np.percentile(fit_list, 5),
                        "95th percentile": np.percentile(fit_list, 95),
                        "env steps": steps
                    })

            if time.time() - start_time > evaluator.report_interval:
                fps, eps = evaluator.report_evals(steps, n_evals)
                if cfg.use_wandb:
                    wandb.log({
                        "evals/sec (EPS)": eps,
                        "fps": fps
                    })


    except KeyboardInterrupt:
        log.debug('Keyboard interrupt detected. Saving final results')

    cm.save_archive(agent_archive, all_actors, n_evals, filename, save_path)
    save_cfg(cfg, save_path)
    evaluator.close_envs()
    log.debug('Done!')