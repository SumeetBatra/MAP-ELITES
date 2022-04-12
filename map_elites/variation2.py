import numpy as np
import torch
import copy

from utils.logger import log
from utils.vectorized import BatchMLP
from utils.signal_slot import EventLoopObject, signal


class VariationOperator(EventLoopObject):
    def __init__(self,
                 cfg,
                 all_actors,
                 elites_map,
                 eval_cache,
                 event_loop,
                 object_id,
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
                 line_sigma=0.05
                 ):
        super().__init__(event_loop, object_id)
        self.cfg = cfg
        self.all_actors = all_actors
        self.elites_map = elites_map
        self.eval_cache = eval_cache
        self.init_mode = False

        # variation hyperparams
        if cfg.crossover_op in ["sbx", "iso_dd"]:
            self.crossover_op = getattr(self, cfg.crossover_op)
        else:
            log.warn("Not using the crossover operator")
            self.crossover_op = False

        if cfg.mutation_op in ['polynomial_mutation', 'gaussian_mutation', 'uniform_mutation']:
            self.mutation_op = getattr(self, cfg.mutation_op)
        else:
            log.warn("Not using the mutation operator")
            self.mutation_op = False
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

    @signal
    def to_evaluate(self): pass

    @signal
    def done(self): pass

    @signal
    def stop(self): pass

    def on_stop(self):
        self.done.emit(self.object_id)

    def init_map(self):
        '''
        Initialize the map of elites
        '''
        rand_init_batch_size = self.cfg.random_init_batch
        log.debug('Initializing the map of elites')
        if len(self.elites_map) <= self.cfg.random_init:
            keys = np.random.randint(len(self.all_actors), size=rand_init_batch_size)
            actor_x_ids = []
            actor_y_ids = [None for _ in range(len(actor_x_ids))]
            if self.mutation_op and not self.crossover_op:
                actor_x_ids = np.random.randint(len(self.all_actors), size=rand_init_batch_size)

            elif self.crossover_op:
                actor_x_ids = np.random.randint(len(self.all_actors), size=rand_init_batch_size)
                actor_y_ids = np.random.randint(len(self.all_actors), size=rand_init_batch_size)

            actors_x_evo = self.all_actors[actor_x_ids][:, 0]
            actors_y_evo = self.all_actors[actor_y_ids][:, 0] if actor_y_ids is not None else None

            device = torch.device('cpu')  # evolve NNs on the cpu, keep gpu memory for batch inference in eval workers
            batch_actors_x = BatchMLP(actors_x_evo, device)
            batch_actors_y = BatchMLP(actors_y_evo, device) if actors_y_evo is not None else None
            with torch.no_grad():
                actors_z = self.evo(batch_actors_x, batch_actors_y, device, self.crossover_op, self.mutation_op)

            # place in eval cache for Evaluator to evaluate
            self.eval_cache[actor_x_ids] = actors_z
            self.to_evaluate.emit(self.object_id, actor_x_ids, self.init_mode)
        # log.debug('Finished elites map initialization!')
        # self.init_mode = False
        # # TODO: better way to do this?
        # self.evolve_batch()


    def flush(self, q):
        '''
        Flush the remaining contents of a q
        :param q: any standard Queue() object
        '''
        while not q.empty():
            q.get_many()

    def evolve_batch(self):
        '''
        Get new batch of agents to evolve
        '''
        batch_size = int(self.cfg['eval_batch_size'] * self.cfg['proportion_evo'])
        # log.debug(f'Evolving a new batch of {batch_size} policies')
        keys = list(self.elites_map.keys())
        actor_x_ids = []
        actor_y_ids = [None for _ in range(len(actor_x_ids))]
        if self.mutation_op and not self.crossover_op:
            actor_x_inds = np.random.randint(len(keys), size=batch_size)
            for i in range(len(actor_x_inds)):
                actor_x_ids += self.elites_map[keys[actor_x_inds[i]]]

        elif self.crossover_op:
            actor_x_inds = np.random.randint(len(keys), size=batch_size)
            actor_y_inds = np.random.randint(len(keys), size=batch_size)
            for i in range(len(actor_x_inds)):
                actor_x_ids.append(self.elites_map[keys[actor_x_inds[i]]][0])
                actor_y_ids.append(self.elites_map[keys[actor_y_inds[i]]][0])

        actors_x_evo = self.all_actors[actor_x_ids][:, 0]
        actors_y_evo = self.all_actors[actor_y_ids][:, 0] if actor_y_ids is not None else None

        device = torch.device('cpu')  # evolve NNs on the cpu, keep gpu memory for batch inference in eval workers
        batch_actors_x = BatchMLP(actors_x_evo, device)
        batch_actors_y = BatchMLP(actors_y_evo, device) if actors_y_evo is not None else None
        with torch.no_grad():
            actors_z = self.evo(batch_actors_x, batch_actors_y, device, self.crossover_op, self.mutation_op)

        # place in eval cache for Evaluator to evaluate
        self.eval_cache[actor_x_ids] = actors_z
        self.to_evaluate.emit(self.object_id, actor_x_ids, self.init_mode)


    def evo(self, actor_x, actor_y, device, crossover_op=None, mutation_op=None):
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
            actor_z_state_dict = self.batch_crossover(actor_x.state_dict(), actor_y.state_dict(), crossover_op, device=device)
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

    def batch_mutation(self, batch_x_state_dict, mutation_op):
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

    def gaussian_mutation(self, x):
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