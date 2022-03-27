'''
Copyright 2019, INRIA
SBX and ido_dd and polynomilal mutauion variation operators based on pymap_elites framework
https://github.com/resibots/pymap_elites/blob/master/map_elites/
pymap_elites main contributor(s):
    Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
    Eloise Dalin , eloise.dalin@inria.fr
    Pierre Desreumaux , pierre.desreumaux@inria.fr
Modified by Olle Nilsson: olle.nilsson19@imperial.ac.uk
'''

import copy
import numpy as np
import torch
import time
from faster_fifo import Queue
from utils.logger import log
from models.bipedal_walker_model import BipedalWalkerNN, model_factory
from torch.multiprocessing import Process as TorchProcess, Pipe, Event
from functools import partial
from pynvml import *
from utils.utils import get_least_busy_gpu
from utils.vectorized import BatchMLP


class VariationOperator(object):
    def __init__(self,
                 cfg,
                 all_actors,
                 all_evolved_actors,
                 elites_map,
                 num_processes=1,
                 num_gpus=0,
                 crossover_op='iso_dd',
                 mutation_op=None,
                 max_gene=False,
                 min_gene=False,
                 mutation_rate=0.05,
                 crossover_rate=0.75,
                 eta_m=5.0,
                 eta_c=10.0,
                 sigma=0.1,
                 max_uniform=0.1,
                 iso_sigma=0.005,
                 line_sigma=0.05):

        self.cfg = cfg
        self.all_actors = all_actors
        self.all_evolved_actors = all_evolved_actors
        self.elites_map = elites_map
        self.num_processes = num_processes
        self.num_gpus = num_gpus
        self.evolve_in_queue = Queue(max_size_bytes=int(1e7))
        self.evolve_out_queue = Queue(max_size_bytes=int(1e7))
        self.remotes, self.locals = zip(*[Pipe() for _ in range(self.num_processes)])

        # evolution hyperparams
        self.crossover_op = True if crossover_op is not None else False
        self.mutation_op = True if mutation_op is not None else False
        self.max = max_gene
        self.min = min_gene
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.eta_m = eta_m
        self.eta_c = eta_c
        self.sigma = sigma
        self.max_uniform = max_uniform
        self.iso_sigma = iso_sigma
        self.line_sigma = line_sigma
        self.evo_cfg = {'min': min_gene, 'max': max_gene, 'mutation_rate': mutation_rate,
                        'crossover_rate': crossover_rate,
                        'eta_m': eta_m, 'eta_c': eta_c, 'sigma': sigma, 'max_uniform': max_uniform,
                        'iso_sigma': iso_sigma,
                        'line_sigma': line_sigma, 'crossover_op': crossover_op, 'mutation_op': mutation_op}

    def init_map(self):
        '''
        Initialize the map of elites
        '''
        rand_init_batch_size = self.cfg['random_init_batch']
        while len(self.elites_map) <= self.cfg['random_init']:
            log.debug(f'Initializing the map of elites with {self.cfg["random_init"]} policies')
            keys = np.random.randint(len(self.all_actors), size=rand_init_batch_size)
            self.evolve_in_queue.put_many(keys.tolist(), block=True, timeout=1e9)

        self.flush(self.evolve_in_queue)

    def flush(self, q):
        '''
        Flush the remaining contents of a q
        :param q: any standard Queue() object
        '''
        while not q.empty():
            q.get_many()



class VariationWorker(object):
    def __init__(self, process_id, all_actors, all_evolved_actors, elites_map, evolve_in_queue, evolve_out_queue, close_processes, remote, num_gpus, evo_cfg):
        self.pid = process_id
        self.all_actors = all_actors
        self.all_evolved_actors = all_evolved_actors
        self.elites_map = elites_map
        self.evolve_in_queue = evolve_in_queue
        self.evolve_out_queue = evolve_out_queue
        self.close_processes = close_processes
        self.remote = remote
        self.num_gpus = num_gpus
        self.terminate = False
        self.evo_cfg = evo_cfg  # hyperparameters for evolution

        log.debug(f'Mutation operator: {evo_cfg["mutation_op"]}')
        log.debug(f'Crossover operator: {evo_cfg["crossover_op"]}')

        if evo_cfg['crossover_op'] in ["sbx", "iso_dd"]:
            self.crossover_op = getattr(self, evo_cfg['crossover_op'])
        else:
            log.warn("Not using the crossover operator")
            self.crossover_op = False

        if evo_cfg['mutation_op'] in ['polynomial_mutation', 'gaussian_mutation', 'uniform_mutation']:
            self.mutation_op = getattr(self, evo_cfg['mutation_op'])
        else:
            log.warn("Not using the mutation operator")
            self.mutation_op = False

        self.process = TorchProcess(target=self._run, daemon=True)
        self.process.start()

    def _run(self):
        if self.num_gpus:
            nvmlInit()  # track available gpu resources

        while not self.terminate:
            actor_keys = self.evolve_in_queue.get_many(block=True, timeout=1e9)
            actors_x_evo, actors_y_evo, actors_z = [], None, []
            if self.mutation_op and not self.crossover_op:
                actors_x_evo = self.all_actors[actor_keys]

            elif self.crossover_op:
                num_keys = len(actor_keys)
                actors_x_keys, actors_y_keys = actor_keys[:num_keys//2], actor_keys[num_keys//2:]
                actors_x_evo, actors_y_evo = self.all_actors[actors_x_keys], self.all_actors[actors_y_keys]

            gpu_id = get_least_busy_gpu(self.num_gpus) if self.num_gpus else 0
            device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
            batch_actors_x = BatchMLP(actors_x_evo, device)
            batch_actors_y = BatchMLP(actors_y_evo, device) if actors_y_evo else None
            actors_z = self.evo(batch_actors_x, batch_actors_y, self.crossover_op, self.mutation_op)

            # update the all_evolved_actors pool and send the keys to the evaluation workers
            # TODO: Fix this
            self.all_evolved_actors[actor_keys] = actors_z

    def _terminate(self):
        self.terminate = True

    def evo(self, actor_x, actor_y, crossover_op=None, mutation_op=None):
        '''
        evolve the agent parameters (in this case, a neural network) using crossover and mutation operators
        crossover needs 2 parents, thus actor_x and actor_y
        mutation can just be performed on actor_x to get actor_z (child)
        '''

        actor_z = copy.deepcopy(actor_x)
        actor_z.type = 'evo'
        actor_z_state_dict = actor_z.state_dict()
        if crossover_op:
            actor_x_ids = actor_x.get_mlp_ids()
            actor_y_ids = actor_y.get_mlp_ids()
            actor_z.set_parent_id(which_parent=1, ids=actor_x_ids)
            actor_z.set_parent_id(which_parent=2, ids=actor_y_ids)
            actor_z_state_dict = self.batch_crossover(actor_x.state_dict(), actor_y.state_dict(), crossover_op)
            if mutation_op:
                actor_z_state_dict = self.batch_mutation(actor_z_state_dict, mutation_op)
        elif mutation_op:
            actor_x_ids = actor_x.get_mlp_ids()
            actor_z.set_parent_id(which_parent=1, ids=actor_x_ids)
            actor_z_state_dict = self.batch_mutation(actor_z_state_dict, mutation_op)

        actor_z.load_state_dict(actor_z_state_dict)
        return actor_z.update_mlps()


    def batch_crossover(self, batch_x_state_dict, batch_y_state_dict, crossover_op, device):
        """
        perform crossover operation b/w two parents (actor x and actor y) to produce child (actor z) for a batch of parents
        :return: a BatchMLP object containing all the actors_z resulting from the crossover of x/y actors
        """
        batch_z_state_dict = copy.deepcopy(batch_x_state_dict)
        for tensor in batch_x_state_dict:
            if 'weight' or 'bias' in tensor:
                batch_z_state_dict[tensor] = crossover_op(batch_x_state_dict[tensor], batch_y_state_dict[tensor]).to(device)
        return batch_z_state_dict


    def batch_mutation(self, batch_x_state_dict, mutation_op, device):
        y = copy.deepcopy(batch_x_state_dict)
        for tensor in batch_x_state_dict:
            if 'weight' or 'bias' in tensor:
                y[tensor] = mutation_op(batch_x_state_dict[tensor])
        return y


    def iso_dd(self, x, y):
        '''
        Iso+Line
        Ref:
        Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
        GECCO 2018
        Support for batch processing
        '''
        with torch.no_grad():
            a = torch.zeros_like(x).normal_(mean=0, std=self.evo_cfg['iso_sigma'])
            b = np.random.normal(0, self.evo_cfg['line_sigma'])
            z = x.clone() + a + b * (y - x)

        if not self.evo_cfg['max'] and not self.evo_cfg['min']:
            return z
        else:
            with torch.no_grad():
                return torch.clamp(z, self.evo_cfg['min'], self.evo_cfg['max'])


    ################################
    # Mutation Operators ###########
    ################################

    def gaussian_mutation(self, x):
        """
        Mutate the params of the agents x according to a gaussian
        """
        with torch.no_grad():
            y = x.clone()
            m = torch.rand_like(y)
            index = torch.where(m < self.evo_cfg['mutation_rate'])
            delta = torch.zeros(index[0].shape).normal_(mean=0, std=self.evo_cfg['sigma']).to(y.device)
            if len(y.shape) == 1:
                y[index[0]] += delta
            elif len(y.shape) == 2:
                y[index[0], index[1]] += delta
            else:
                # 3D i.e. BatchMLP
                y[index[0], index[1], index[2]] += delta
            if not self.evo_cfg['max'] and not self.evo_cfg['min']:
                return y
            else:
                return torch.clamp(y, self.evo_cfg['min'], self.evo_cfg['max'])


