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

import math
import numpy as np
import multiprocessing

# from scipy.spatial import cKDTree : TODO -- faster?
import wandb
import time
import os
import shutil
import glob
import torch
from sklearn.neighbors import KDTree
from models.bipedal_walker_model import model_factory
from itertools import count

from map_elites import common as cm
from utils.logger import log, config_wandb
from utils.utils import *


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


# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def __evaluate(t):
    z, f = t  # evaluate z with function f
    fit, desc = f(z)
    return cm.Species(z, desc, fit)


# map-elites algorithm (CVT variant)
def compute_nn(cfg,
               envs,
               variation_operator,
               actors_file,
               filename,
               save_path,
               n_niches=1000,
               max_evals=1e5):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile
        dim_map: dimensionality of the map. Ex. % time contact w/ ground of 4 legs = 4 dims, etc
        dim_x: dimensionality of the behavior descriptor
        f: the environment which returns a fitness and behavior descriptor
    """

    num_gpus = cfg['num_gpus']
    # create the CVT
    cluster_centers = cm.cvt(n_niches, cfg['dim_map'],
                             cfg['cvt_samples'], cfg['cvt_use_cache'])
    kdt = KDTree(cluster_centers, leaf_size=30, metric='euclidean')
    cm.__write_centroids(cluster_centers)

    archive = {}  # init archive (empty)
    n_evals = 0  # number of evaluations since the beginning
    b_evals = 0  # number evaluation since the last dump
    cp_evals = 0 # number of evaluations since last checkpoint dump
    steps = 0  # env steps
    gpu_id = 0

    # main loop
    try:
        while (n_evals < max_evals):
            start_time = time.time()
            to_evaluate = []
            # random initialization
            if len(archive) <= cfg['random_init']:  # initialize a |random_init| number of actors
                log.debug("Initializing the neural network actors' weights from scratch")
                for i in range(0, cfg['random_init_batch'] + 1):
                    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
                    actor = model_factory(hidden_size=128, device=device).to(device)
                    log.debug(f'New actor going to gpu {gpu_id}')
                    to_evaluate += [actor]
                    gpu_id = (gpu_id + 1) % num_gpus
            else:  # variation/selection loop
                log.debug("Selection/Variation loop of existing actors")
                # copy and add variation
                to_evaluate += variation_operator(archive, cfg['eval_batch_size'], cfg['proportion_evo'])

            log.debug(f"Evaluating {len(to_evaluate)} policies")

            # evaluations of the fitness and BD of new batch
            solutions = envs.eval_policy(to_evaluate)
            frames = sum(sol[1] for sol in solutions)
            steps += frames

            n_evals += len(to_evaluate)
            b_evals += len(to_evaluate)
            cp_evals += len(to_evaluate)
            log.debug(f'{n_evals/int(cfg["max_evals"])}')
            # add to archive
            for idx, solution in enumerate(solutions):
                # TODO: need to check for if robot is alive?
                agent = Individual(genotype=to_evaluate[idx], phenotype=solution[2], fitness=solution[0])
                added = __add_to_archive(agent, agent.phenotype, archive, kdt)
                if added:
                    actors_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(n_evals,
                                                                           agent.genotype.id,
                                                                           agent.fitness,
                                                                           str(agent.phenotype).strip("[]"),
                                                                           str(agent.centroid).strip("()"),
                                                                           agent.genotype.parent_1_id,
                                                                           agent.genotype.parent_2_id,
                                                                           agent.genotype.type,
                                                                           agent.genotype.novel,
                                                                           agent.genotype.delta_f))
                    actors_file.flush()

            # maybe save a checkpoint
            if cp_evals >= cfg['cp_save_period'] and cfg['cp_save_period'] != -1:
                save_checkpoint(archive, n_evals, filename, cfg['checkpoint_dir'], cfg)
                while len(get_checkpoints(cfg['checkpoint_dir'])) > cfg['keep_checkpoints']:
                    oldest_checkpoint = get_checkpoints(cfg['checkpoint_dir'])[0]
                    if os.path.exists(oldest_checkpoint):
                        log.debug('Removing %s', oldest_checkpoint)
                        shutil.rmtree(oldest_checkpoint)
                cp_evals = 0

            eps = round(len(to_evaluate) / (time.time() - start_time), 1)
            fps = round(frames / (time.time() - start_time), 1)

            # logging
            fit_list = np.array([x.fitness for x in archive.values()])
            # write log
            log.info(f'n_evals: {n_evals}, mean fitness: {np.mean(fit_list)}, median fitness: {np.median(fit_list)}, \
                5th percentile: {np.percentile(fit_list, 5)}, 95th percentile: {np.percentile(fit_list, 95)}')
            log.debug(f'Evals/sec (EPS): {eps}, FPS: {fps}, steps: {steps}')
            wandb.log({
                "evals": n_evals,
                "mean fitness": np.mean(fit_list),
                "median fitness": np.median(fit_list),
                "5th percentile": np.percentile(fit_list, 5),
                "95th percentile": np.percentile(fit_list, 95),
                "evals/sec (Eps)": eps,
                "fps": fps,
                "env steps": steps
            })
    except KeyboardInterrupt:
        log.debug('Keyboard interrupt detected. Saving final results')

    cm.save_archive(archive, n_evals, filename, save_path, save_models=True)
    save_cfg(cfg, save_path)
    envs.close()


class Individual(object):
    _ids = count(0)

    def __init__(self, genotype, phenotype, fitness, centroid=None):
        """
        A single agent
        param genotype:  The parameters that produced the behavior. I.e. neural network, etc.
        param phenotype: the resultant behavior i.e. the behavioral descriptor
        param fitness: the fitness of the model. In the case of a neural network, this is the total accumulated rewards
        param centroid: the closest CVT generator (a behavior) to the behavior that this individual exhibits
        """
        genotype.id = next(self._ids)
        Individual.current_id = genotype.id  # TODO: not sure what this is for
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness
        self.centroid = centroid
        self.novelty = None
