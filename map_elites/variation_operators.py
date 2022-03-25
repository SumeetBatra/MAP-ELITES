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
from models.bipedal_walker_model import BipedalWalkerNN
from torch.multiprocessing import Process as TorchProcess, Pipe, Event
from functools import partial
from pynvml import *
from utils.utils import get_least_busy_gpu
from utils.vectorized import BatchMLP


def parallel_variation_worker(process_id,
                              var_in_queue,
                              var_out_queue,
                              close_processes,
                              remote,
                              num_gpus):
    if num_gpus:
        nvmlInit()
    while True:
        try:
            # try to mutate/crossover a batch of policies
            try:
                pass
            except BaseException:
                pass
            if close_processes.set():
                log.debug(f'Close Variation Worker Process ID {process_id}')
                remote.send(process_id)
                time.sleep(5)
                break
        except KeyboardInterrupt:
            break


class VariationWorker(object):
    def __init__(self, process_id, var_in_queue, var_out_queue, close_processes, remote, num_gpus, evo_cfg):
        self.pid = process_id
        self.var_in_queue = var_in_queue
        self.var_out_queue = var_out_queue
        self.close_processes = close_processes
        self.remote = remote
        self.num_gpus = num_gpus
        self.terminate = False
        self.evo_cfg = evo_cfg  # hyperparameters for evolution

        if num_gpus:
            nvmlInit()

        self.process = TorchProcess(target=self._run, daemon=True)
        self.process.start()

    def _run(self):
        with torch.no_grad():
            while not self.terminate:
                try:
                    try:
                        # try to mutate/crossover a batch of policies
                        actors_x, actors_y, crossover_op, mutation_op = self.var_in_queue.get_nowait()
                        actors_z = self.batch_evo(actors_x, actors_y, crossover_op, mutation_op)
                        self.var_out_queue.put_many(actors_z, block=True, timeout=1e9)
                    except BaseException:
                        pass
                    if self.close_processes.set():
                        log.debug(f'Close Variation Worker Process ID {self.pid}')
                        self.remote.send(self.pid)
                        time.sleep(5)
                        break
                except KeyboardInterrupt:
                    break

    def batch_evo(self, actors_x, actors_y=None, crossover_op=None, mutation_op=None):
        '''
        evolve the agent parameters (in this case, a neural network) using crossover and mutation operators
        crossover needs 2 parents, thus actor_x and actor_y
        mutation can just be performed on actor_x to get actor_z (child)
        '''
        if self.num_gpus:
            gpu_id = get_least_busy_gpu(self.num_gpus)
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        batch_actors_x = BatchMLP(actors_x, device)
        batch_actors_y = BatchMLP(actors_y, device)

        actors_z = [copy.deepcopy(actor_x) for actor_x in actors_x]
        batch_actors_z = BatchMLP(actors_z, device)
        actors_z_state_dict = batch_actors_z.state_dict()
        for actor_z in actors_z:
            actor_z.type = 'evo'

        if crossover_op:
            for x, y, z in zip(actors_x, actors_y, actors_z):
                z.parent_1_id = x.id
                z.parent_2_id = y.id
            actors_z_state_dict = self.batch_crossover(batch_actors_x.state_dict(), batch_actors_y.state_dict(), crossover_op, device)
            if mutation_op:
                actors_z_state_dict = self.batch_mutation(actors_z_state_dict, mutation_op, device)
            elif mutation_op:
                for actor_x, actor_z in zip(actors_x, actors_z):
                    actor_z.parent_1_id = actor_x.id
                    actor_z.parent_2_id = None
                actors_z_state_dict = self.batch_mutation(actors_z_state_dict, mutation_op, device)
        batch_actors_z.load_state_dict(actors_z_state_dict)
        actors_z = batch_actors_z.update_mlps()
        return actors_z

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

    def batch_iso_dd(self, x, y):
        '''
        Iso+Line
        Ref:
        Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
        GECCO 2018
        Support for batch processing
        '''
        with torch.no_grad():
            a = torch.zeros_like(x).normal_(mean=0, std=self.iso_sigma)
            b = np.random.normal(0, self.line_sigma)
            z = x.clone() + a + b * (y - x)

        if not self.max and not self.min:
            return z
        else:
            with torch.no_grad():
                return torch.clamp(z, self.min, self.max)


    ################################
    # Mutation Operators ###########
    ################################

    def batch_gaussian_mutation(self, x):
        """
        Mutate the params of the agents x according to a gaussian
        """
        with torch.no_grad():
            y = x.clone()
            m = torch.rand_like(y)
            index = torch.where(m < self.mutation_rate)
            delta = torch.zeros(index[0].shape).normal_(mean=0, std=self.sigma).to(y.device)
            if len(y.shape) == 1:
                y[index[0]] += delta
            elif len(y.shape) == 2:
                y[index[0], index[1]] += delta
            else:
                # 3D i.e. BatchMLP
                y[index[0], index[1], index[2]] += delta
            if not self.max and not self.min:
                return y
            else:
                return torch.clamp(y, self.min, self.max)


class VariationOperator(object):
    """
    A class for applying the variation operator in parallel.
    """
    def __init__(self,
                 num_cpu = 1,
                 num_gpu = 1,
                 crossover_op = 'iso_dd',
                 mutation_op = None,
                 max_gene = False,
                 min_gene = False,
                 mutation_rate = 0.05,
                 crossover_rate = 0.75,
                 eta_m = 5.0,
                 eta_c = 10.0,
                 sigma = 0.1,
                 max_uniform = 0.1,
                 iso_sigma = 0.005,
                 line_sigma = 0.05):

        if crossover_op in ["sbx", "iso_dd"]:
            self.crossover_op = getattr(self, crossover_op)
        else:
            log.warn("Not using the crossover operator")
            self.crossover_op = False

        if mutation_op in ['polynomial_mutation', 'gaussian_mutation', 'uniform_mutation']:
            self.mutation_op = getattr(self, mutation_op)
        else:
            log.warn("Not using the mutation operator")
            self.mutation_op = False
        log.debug(f'Mutation operator: {self.mutation_op}')
        log.debug(f'Crossover operator: {self.crossover_op}')

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
        self.evo_cfg = {'min': min_gene, 'max': max_gene, 'mutation_rate': mutation_rate, 'crossover_rate': crossover_rate,
                        'eta_m': eta_m, 'eta_c': eta_c, 'sigma': sigma, 'max_uniform': max_uniform, 'iso_sigma': iso_sigma,
                        'line_sigma': line_sigma}

        self.n_processes = num_cpu
        self.num_gpu = num_gpu
        self.var_in_queue = Queue()
        self.var_out_queue = Queue()
        self.remotes, self.locals = zip(*[Pipe() for _ in range(self.n_processes + 1)])
        self.close_processes = Event()

        self.processes = [VariationWorker(process_id,
                                          self.var_in_queue,
                                          self.var_out_queue,
                                          self.close_processes,
                                          self.remotes[process_id],
                                          self.num_gpu,
                                          self.evo_cfg) for process_id in range(self.n_processes)]

    def get_new_batch(self, archive, batch_size, proportion_evo):
        '''
        Get a new batch of policies to evolve
        Args:
            archive: archive of elites
            batch_size: number of policies to process in a batch
            proportion_evo: proportion of sampled policies to evolve
        '''
        keys = list(archive.keys)
        actors_x_evo, actors_y_evo = [], None
        actors_z = []
        # sample from archive
        if self.mutation_op and not self.crossover_op:
            # TODO: this can be optimized with torch Dataset class potentially
            actors_x_evo = []
            rand_evo = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
            for n in range(0, len(rand_evo)):
                actors_x_evo += [archive[keys[rand_evo[n]]]]

        elif self.crossover_op:
            actors_x_evo = []
            actors_y_evo = []
            rand_evo_1 = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
            rand_evo_2 = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
            for n in range(0, len(rand_evo_1)):
                actors_x_evo += [archive[keys[rand_evo_1[n]]]]
                actors_y_evo += [archive[keys[rand_evo_2[n]]]]

        self.var_in_queue.put(actors_x_evo, actors_y_evo if actors_y_evo else None, self.crossover_op, self.mutation_op, block=True, timeout=1e9)


    def __call__(self, archive, batch_size, proportion_evo):
        '''
        the variation operator object is called to apply the varation

        Parameters:
            archive (dict): main archive
            batch_size (int): how many actors to sample per generation
            proportion_evo (float): proportion of GA variation
            critic: Critic to use in gradient variation
            states: states (tansitions) to use in policy gradient update
            train_batch_size (int): batch size for policy gradient update
            nr_of_steps_act (int): nr of gradient steps to take per actor
        '''

        keys = list(archive.keys())
        actors_z = []
        # sample from archive
        if self.mutation_op and not self.crossover_op:
            # TODO: this can be optimized with torch Dataset class potentially
            actors_x_evo = []
            rand_evo = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
            for n in range(0, len(rand_evo)):
                actors_x_evo += [archive[keys[rand_evo[n]]]]

        elif self.crossover_op:
            actors_x_evo = []
            actors_y_evo = []
            rand_evo_1 = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
            rand_evo_2 = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
            for n in range(0, len(rand_evo_1)):
                actors_x_evo += [archive[keys[rand_evo_1[n]]]]
                actors_y_evo += [archive[keys[rand_evo_2[n]]]]

        # num_evos = int(batch_size * proportion_evo)
        # num_evos_per_worker = num_evos // self.n_processes
        # if self.crossover_op and self.mutation_op:
        #     evo_fn = partial(self.evo, actors_x_evo[n].genotype, actors_y_evo[n].genotype, self.crossover_op, self.mutation_op)
        # elif self.crossover_op and not self.mutation_op:
        #     pass



        # apply GA variation
        for n in range(len(actors_x_evo)):
            if self.crossover_op:
                if self.mutation_op:
                    actors_z += [self.evo(actors_x_evo[n].genotype, actors_y_evo[n].genotype, self.crossover_op, self.mutation_op)]
                else:
                    actors_z += [self.evo(actors_x_evo[n].genotype, actors_y_evo[n].genotype, self.crossover_op)]
            elif self.mutation_op:
                actors_z += [self.evo(actors_x_evo[n].genotype, False, False, self.mutation_op)]

        return actors_z


    def evo(self, actor_x, actor_y=None, crossover_op=None, mutation_op=None):
        '''
        evolve the agent parameters (in this case, a neural network) using crossover and mutation operators
        crossover needs 2 parents, thus actor_x and actor_y
        mutation can just be performed on actor_x to get actor_z (child)
        '''
        actor_z = copy.deepcopy(actor_x)
        actor_z.type = 'evo'
        actor_z_state_dict = actor_z.state_dict()
        if crossover_op:
            actor_z.parent_1_id = actor_x.id
            actor_z.parent_2_id = actor_y.id
            actor_z_state_dict = self.crossover(actor_x.state_dict(), actor_y.state_dict(), crossover_op)
            if mutation_op:
                actor_z_state_dict = self.mutation(actor_z_state_dict, mutation_op)
        elif mutation_op:
            actor_z.parent_1_id = actor_x.id
            actor_z.parent_2_id = None
            actor_z_state_dict = self.mutation(actor_x.state_dict(), mutation_op)
        actor_z.load_state_dict(actor_z_state_dict)
        return actor_z

    def crossover(self, actor_x_state_dict, actor_y_state_dict, crossover_op):
        """
        perform crossover operation b/w two parents (actor x and actor y) to produce child (actor z)
        """
        actor_z_state_dict = copy.deepcopy(actor_x_state_dict)
        for tensor in actor_x_state_dict:
            if 'weight' or 'bias' in tensor:
                actor_z_state_dict[tensor] = crossover_op(actor_x_state_dict[tensor], actor_y_state_dict[tensor]).to(actor_x_state_dict[tensor].device)
        return actor_z_state_dict

    def mutation(self, actor_x_state_dict, mutation_op):
        y = copy.deepcopy(actor_x_state_dict)
        for tensor in actor_x_state_dict:
            if 'weight' or 'bias' in tensor:
                y[tensor] = mutation_op(actor_x_state_dict[tensor])
        return y

    ################################
    # Crossover Operators ###########
    ################################

    def iso_dd(self, x, y):
        '''
        Iso+Line
        Ref:
        Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
        GECCO 2018
        '''
        if x.device != y.device:  # tensors need to be on the same gpu (or both on cpu)
            x = x.cpu()
            y = y.cpu()
        with torch.no_grad():
            a = torch.zeros_like(x).normal_(mean=0, std=self.iso_sigma)
            b = np.random.normal(0, self.line_sigma)
            z = x.clone() + a + b * (y - x)

        if not self.max and not self.min:
            return z
        else:
            with torch.no_grad():
                return torch.clamp(z, self.min, self.max)

    def sbx(self, x, y):
        if not self.max and not self.min:
            return self.__sbx_unbounded(x, y)
        else:
            return self.__sbx_bounded(x, y)

    def __sbx_unbounded(self, x, y):
        '''
        SBX (cf Deb 2001, p 113) Simulated Binary Crossover
        Unbounded version
        A large value ef eta gives a higher probablitity for
        creating a `near-parent' solutions and a small value allows
        distant solutions to be selected as offspring.
        '''
        with torch.no_grad():
            z = x.clone()
            c = torch.rand_like(z)
            index = torch.where(c < self.crossover_rate)
            r1 = torch.rand(index[0].shape)
            r2 = torch.rand(index[0].shape)

            if len(z.shape) == 1:
                diff = torch.abs(x[index[0]] - y[index[0]])
                x1 = torch.min(x[index[0]], y[index[0]])
                x2 = torch.max(x[index[0]], y[index[0]])
                z_idx = z[index[0]]
            else:
                diff = torch.abs(x[index[0], index[1]] - y[index[0], index[1]])
                x1 = torch.min(x[index[0], index[1]], y[index[0], index[1]])
                x2 = torch.max(x[index[0], index[1]], y[index[0], index[1]])
                z_idx = z[index[0], index[1]]

            beta_q = torch.where(r1 <= 0.5, (2.0 * r1) ** (1.0 / (self.eta_c + 1)), (1.0 / (2.0 * (1.0 - r1))) ** (1.0 / (self.eta_c + 1)))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            z_mut = torch.where(diff > 1e-15, torch.where(r2 <= 0.5, c2, c1), z_idx)

            if len(y.shape) == 1:
                z[index[0]] = z_mut
            else:
                z[index[0], index[1]] = z_mut
            return z

    def __sbx_bounded(self, x, y):
        '''
        SBX (cf Deb 2001, p 113) Simulated Binary Crossover
        A large value ef eta gives a higher probablitity for
        creating a `near-parent' solutions and a small value allows
        distant solutions to be selected as offspring.
        '''
        with torch.no_grad():
            z = x.clone()
            c = torch.rand_like(z)
            index = torch.where(c < self.crossover_rate)
            r1 = torch.rand(index[0].shape)
            r2 = torch.rand(index[0].shape)

            if len(z.shape) == 1:
                diff = torch.abs(x[index[0]] - y[index[0]])
                x1 = torch.min(x[index[0]], y[index[0]])
                x2 = torch.max(x[index[0]], y[index[0]])
                z_idx = z[index[0]]
            else:
                diff = torch.abs(x[index[0], index[1]] - y[index[0], index[1]])
                x1 = torch.min(x[index[0], index[1]], y[index[0], index[1]])
                x2 = torch.max(x[index[0], index[1]], y[index[0], index[1]])
                z_idx = z[index[0], index[1]]


            beta = 1.0 + (2.0 * (x1 - self.min) / (x2 - x1))
            alpha = 2.0 - beta ** - (self.eta_c + 1)
            beta_q = torch.where(r1 <= (1.0 / alpha), (r1 * alpha) ** (1.0 / (self.eta_c + 1)), (1.0 / (2.0 - r1 * alpha)) ** (1.0 / (self.eta_c + 1)))

            c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

            beta = 1.0 + (2.0 * (self.max - x2) / (x2 - x1))
            alpha = 2.0 - beta ** - (self.eta_c + 1)

            beta_q = torch.where(r1 <= (1.0 / alpha), (r1 * alpha) ** (1.0 / (self.eta_c + 1)), (1.0 / (2.0 - r1 * alpha)) ** (1.0 / (self.eta_c + 1)))
            c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

            c1 = torch.clamp(c1, self.min, self.max)
            c2 = torch.clamp(c2, self.min, self.max)

            z_mut = torch.where(diff > 1e-15, torch.where(r2 <= 0.5, c2, c1), z_idx)

            if len(y.shape) == 1:
                z[index[0]] = z_mut
            else:
                z[index[0], index[1]] = z_mut
            return z

################################
# Mutation Operators ###########
################################

    def polynomial_mutation(self, x):
        '''
        Cf Deb 2001, p 124 ; param: eta_m
        mutate the params of the agent according to some polynomial
        '''
        with torch.no_grad():
            y = x.clone()
            m = torch.rand_like(y)
            index = torch.where(m < self.mutation_rate)
            r = torch.rand(index[0].shape)
            delta = torch.where(r < 0.5,\
                (2 * r) ** (1.0 / (self.eta_m + 1.0)) -1.0,\
                        1.0 - ((2.0 * (1.0 - r)) ** (1.0 / (self.eta_m + 1.0))))
            if len(y.shape) == 1:
                y[index[0]] += delta
            else:
                y[index[0], index[1]] += delta

            if not self.max and not self.min:
                return y
            else:
                return torch.clamp(y, self.min, self.max)

    def gaussian_mutation(self, x):
        """
        Mutate the params of the agent x according to a gaussian
        """
        with torch.no_grad():
            y = x.clone()
            m = torch.rand_like(y)
            index = torch.where(m < self.mutation_rate)
            delta = torch.zeros(index[0].shape).normal_(mean=0, std=self.sigma).to(y.device)
            if len(y.shape) == 1:
                y[index[0]] += delta
            else:
                y[index[0], index[1]] += delta

            if not self.max and not self.min:
                return y
            else:
                return torch.clamp(y, self.min, self.max)

    def uniform_mutation(self, x):
        """
        Mutate the params of the agent x according to a uniform distribution
        """
        with torch.no_grad():
            y = x.clone()
            m = torch.rand_like(y)
            index = torch.where(m < self.mutation_rate)
            delta = torch.zeros(index[0].shape).uniform_(-self.max_uniform, self.max_uniform)
            if len(y.shape) == 1:
                y[index[0]] += delta
            else:
                y[index[0], index[1]] += delta

            if not self.max and not self.min:
                return y
            else:
                return torch.clamp(y, self.min, self.max)


if __name__ == "__main__":
    actor_x = BipedalWalkerNN(input_dim=5, hidden_size=5, action_dim=1)
    actor_y = BipedalWalkerNN(input_dim=5, hidden_size=5, action_dim=1)
    var = VariationOperator(mutation_op=True, mutation_rate=0.95)
    actor_z = var.evo(actor_x, actor_y, mutation_op=var.gaussian_mutation, crossover_op=var.iso_dd)
    print(actor_x.state_dict()["layers.0.bias"])
    print(actor_y.state_dict()["layers.0.bias"])
    print(actor_z.state_dict()["layers.0.bias"])