import time

from numpy import ndarray
from map_elites import common as cm
from map_elites.evaluator import Evaluator, MAPPED
from map_elites.variation import VariationOperator
from utils.signal_slot import EventLoopObject, signal, Timer
from utils.logger import log
from models.policy import Policy


class Trainer(EventLoopObject):
    def __init__(self, cfg, all_actors, elites_map, kdt, mutator, evaluator, event_loop, object_id):
        super().__init__(event_loop, object_id)
        self.cfg = cfg
        self.all_actors = all_actors
        self.elites_map = elites_map
        self.kdt = kdt
        self.unused_keys = list(range(len(self.all_actors)))

        self.mutator: VariationOperator = mutator
        self.evaluator: Evaluator = evaluator

        self.num_policies = cfg.random_init_batch
        self.step_size = 64  # number of new policies added to the archive before increasing the number of parallel environments
        self.num_envs = cfg.num_agents
        self.init_mode = True

        def periodic(period, callback):
            return Timer(self.event_loop, period).timeout.connect(callback)


    @signal
    def stop(self): pass

    @signal
    def release_keys(self): pass

    @signal
    def eval_results(self): pass

    @signal
    def init_success(self): pass

    def on_release_keys(self, mapped_actor_keys):
        self.mutator.on_release(mapped_actor_keys)

    def on_start(self):
        self.evaluator.init_env()
        self.init_success.emit()

    def on_stop(self):
        self.evaluator.on_stop()

    def periodic(self, period, callback):
        return Timer(self.event_loop, period).timeout.connect(callback)

    def train(self):
        # mutate batch of policies if enough free policies available
        eval_res = None
        batch_size = self.cfg.random_init_batch if self.init_mode else self.num_policies
        var_res = self.mutator.maybe_mutate_new_batch(batch_size, self.init_mode)
        (mutated_policies, mutated_policy_keys) = var_res if var_res is not None else (None, None)

        # try to evaluate a batch of policies
        if mutated_policies is not None:
            eval_res = self.evaluator.evaluate_batch(mutated_policies, mutated_policy_keys)
        # if evaluation was successful, get metadata and map the evaluated policies into the archive
        (agents, mutated_actor_keys, frames, runtime, avg_ep_length) = eval_res if eval_res is not None else \
            (None, None, None, None, None)

        if agents is not None:
            metadata, evals = self._map_agents(agents, mutated_actor_keys, mutated_policies)
            self.eval_results.emit(self.object_id, metadata, evals, frames, runtime, avg_ep_length)
            self.mutator.update_eval_queue(agents)
        else:  # evaluation was not successful for some reason, release the keys
            mutated_actor_keys = eval_res
            self.mutator.on_release(mutated_actor_keys)

    def _map_agents(self, agents, evaluated_actors_keys, mutated_policies):
        '''
        Map the evaluated agents using their behavior descriptors. Send the metadata back to the main process for logging
        :param behav_descs: behavior descriptors of a batch of evaluated agents
        :param rews: fitness scores of a batch of evaluated agents
        '''
        start_time = time.time()
        metadata = []
        evals = len(agents)

        for policy, agent in zip(mutated_policies, agents):
            added = False
            niche_index = self.kdt.query([agent.phenotype], k=1)[1][0][0]  # get the closest voronoi cell to the behavior descriptor
            niche = self.kdt.data[niche_index]
            n = cm.make_hashable(niche)
            agent.centroid = n
            if n in self.elites_map:
                # natural selection
                map_agent_id, fitness = self.elites_map[
                    n]  # TODO: make sure this interface is correctly implemented
                if agent.fitness > fitness:
                    self.elites_map[n] = (map_agent_id, agent.fitness)
                    agent.genotype = map_agent_id
                    # override the existing agent in the actors pool @ map_agent_id. This species goes extinct b/c a more fit one was found
                    stored_actor, _ = self.all_actors[map_agent_id]
                    stored_actor.load_state_dict(policy.state_dict())
                    self.all_actors[map_agent_id] = (stored_actor, MAPPED)
                    added = True
            else:
                # need to find a new, unused agent id since this agent maps to a new cell
                agent_id = self._find_available_agent_id()
                self.elites_map[n] = (agent_id, agent.fitness)
                agent.genotype = agent_id
                stored_actor, _ = self.all_actors[agent_id]
                stored_actor.load_state_dict(policy.state_dict())
                self.all_actors[agent_id] = (stored_actor, MAPPED)
                added = True

            if added:
                metadata.append(agent)
        runtime = time.time() - start_time
        log.debug(f'Finished mapping elite agents in {runtime:.1f} seconds')
        self.mutator.on_release(evaluated_actors_keys)
        return metadata, evals

    def _find_available_agent_id(self):
        '''
        If an agent maps to a new cell in the elites map, need to give it a new, unused id from the pool of agents
        :returns an agent id that's not being used by some other agent in the elites map
        '''
        agent_id = self.unused_keys[0]
        del self.unused_keys[0]
        return agent_id

    def get_components(self):
        return self.mutator, self.evaluator

    def maybe_resize_vec_env(self):
        init_mode = True if len(self.elites_map) <= self.cfg.random_init else False
        if not init_mode and self.init_mode:
            # one time resizing of vec envs to be size of elites map instead of init_batch_size
            self.init_mode = False
            self.num_envs = len(self.elites_map) * self.cfg.mutations_per_policy * self.cfg.num_envs_per_policy
            self.num_policies = len(self.elites_map)
            self.evaluator.resize_env(self.num_envs)

        elif len(self.all_actors) >= len(self.elites_map) >= self.step_size + self.num_policies:
            log.debug(f'Elites map size is now {len(self.elites_map)}')
            log.debug(f'Resizing the number of envs from {self.cfg.mutations_per_policy * self.num_envs * self.cfg.num_envs_per_policy} to '
                      f'{self.cfg.mutations_per_policy * len(self.elites_map) * self.cfg.num_envs_per_policy}')
            self.num_envs = len(self.elites_map) * self.cfg.mutations_per_policy * self.cfg.num_envs_per_policy
            self.num_policies = len(self.elites_map)
            self.evaluator.resize_env(self.num_envs)
