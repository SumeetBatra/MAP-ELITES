import time
import os
import json
import numpy as np
import wandb

from collections import deque
from utils.signal_slot import EventLoop, EventLoopObject, EventLoopProcess, signal, Timer
from utils.logger import log
from utils.utils import cfg_dict
from map_elites import common as cm
from map_elites.evaluator import Individual, UNUSED, MAPPED


class Runner(EventLoopObject):
    def __init__(self, cfg, archive, elites_map, eval_cache, all_actors, kdt, actors_file, filename, unique_name='runner loop'):
        '''
        This is the main loop that starts other event loops and processes file i/o, logging, etc
        '''
        self.cfg = cfg
        self.event_loop = EventLoop(unique_loop_name=f'{unique_name}_EvtLoop')
        super().__init__(self.event_loop, unique_name)

        self.stopped = False

        # map elites objects
        self.archive = archive
        self.elites_map = elites_map
        self.eval_cache = eval_cache
        self.all_actors = all_actors
        self.kdt = kdt
        self.checkpoints_dir = self.cfg.checkpoint_dir
        self.actors_file = actors_file
        self.filename = filename

        # hyperparams
        self.max_evals = cfg.max_evals

        # other event loops
        self.variation_op, self.evaluator = None, None

        # logging variables
        self.avg_stats_intervals = (2, 12, 60)  # 10 seconds, 1 minute, 5 minutes
        self._report_interval = 5.0  # report every 5 seconds
        self._last_report = 0.0
        self.eval_stats = deque([], maxlen=max(self.avg_stats_intervals))
        self.total_env_steps, self.total_evals = 0, 0

        # timers for periodic logging / saving results
        def periodic(period, callback):
            return Timer(self.event_loop, period).timeout.connect(callback)

        periodic(self._report_interval, self.report_evals)
        periodic(self.cfg.cp_save_period_sec, self.save_checkpoint)

    @signal
    def stop(self): pass

    @property
    def report_interval(self):
        return self._report_interval

    @property
    def last_report(self):
        return self._last_report

    def on_eval_results(self, oid, agents, evaluated_actors_keys, frames):
        metadata, evals = self._map_agents(agents, evaluated_actors_keys)
        self._update_training_stats(frames, evals)
        fit_list = []
        for agent in metadata:
            fit_list.append(agent.fitness)
            self.archive.append(agent)
            self.actors_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(self.total_evals,
                                                                       agent.genotype_id,
                                                                       agent.fitness,
                                                                       agent.phenotype,
                                                                       agent.centroid,
                                                                       agent.parent_1_id,
                                                                       agent.parent_2_id,
                                                                       agent.genotype_type,
                                                                       agent.genotype_novel,
                                                                       agent.genotype_delta_f))
            self.actors_file.flush()
            self._log_metadata(fit_list)
            self._maybe_stop_training()

    def _log_metadata(self, fit_list):
        # write log
        log.info(f'n_evals: {self.total_evals}, mean fitness: {np.mean(fit_list)}, median fitness: {np.median(fit_list)}, \
            5th percentile: {np.percentile(fit_list, 5)}, 95th percentile: {np.percentile(fit_list, 95)}')
        fit_list = np.array(fit_list)
        if self.cfg['use_wandb']:
            wandb.log({
                "evals": self.total_evals,
                "mean fitness": np.mean(fit_list),
                "median fitness": np.median(fit_list),
                "5th percentile": np.percentile(fit_list, 5),
                "95th percentile": np.percentile(fit_list, 95),
                "env steps": self.total_env_steps
            })


    def save_checkpoint(self):
        checkpoint_name = f'checkpoint_{self.total_evals:09d}/'
        filepath = os.path.join(self.checkpoints_dir, checkpoint_name)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self._save_archive(self.archive, self.all_actors, self.total_evals, self.filename, save_path=filepath, save_models=True)
        self._save_cfg(self.cfg, filepath)

    def _find_available_agent_id(self):
        '''
        If an agent maps to a new cell in the elites map, need to give it a new, unused id from the pool of agents
        :returns an agent id that's not being used by some other agent in the elites map
        '''
        return next(i for i in range(len(self.all_actors)) if self.all_actors[i][1] == UNUSED)

    def _map_agents(self, agents, evaluated_actors_keys):
        '''
        Map the evaluated agents using their behavior descriptors. Send the metadata back to the main process for logging
        :param behav_descs: behavior descriptors of a batch of evaluated agents
        :param rews: fitness scores of a batch of evaluated agents
        '''
        start_time = time.time()
        metadata = []
        evals = len(agents)
        policies = self.eval_cache[evaluated_actors_keys]
        for policy, agent in zip(policies, agents):
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
                    self.all_actors[map_agent_id] = (policy, MAPPED)
                    added = True
            else:
                # need to find a new, unused agent id since this agent maps to a new cell
                agent_id = self._find_available_agent_id()
                self.elites_map[n] = (agent_id, agent.fitness)
                agent.genotype = agent_id
                self.all_actors[agent_id] = (policy, MAPPED)
                added = True

            if added:
                md = (
                agent.genotype_id, agent.fitness, str(agent.phenotype).strip("[]"), str(agent.centroid).strip("()"),
                agent.parent_1_id, agent.parent_2_id, agent.genotype_type, agent.genotype_novel,
                agent.genotype_delta_f)
                metadata.append(agent)
        runtime = time.time() - start_time
        log.debug(f'Finished mapping elite agents in {runtime:.1f} seconds')

        return metadata, evals

    def _save_archive(self, archive, all_actors, gen, archive_name, save_path, save_models=False):
        def write_array(a, f):
            for i in a:
                f.write(str(i) + ' ')

        filename = f"{save_path}/archive_{archive_name}_" + str(gen) + '.dat'
        model_path = save_path + '/policies/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        with open(filename, 'w') as f:
            for k in archive:
                f.write(str(k.fitness) + ' ')
                write_array(k.centroid, f)
                write_array(k.phenotype, f)
                f.write(str(k.genotype_id) + ' ')
                f.write("\n")
                if save_models:
                    all_actors[k.genotype][0].save(model_path + archive_name + '_actor_' + str(k.genotype_id) + '.pt')

    def _save_cfg(self, cfg, save_path):
        cfg = cfg_dict(cfg)
        cfg_file = os.path.join(save_path, 'cfg.json')
        with open(cfg_file, 'w') as json_file:
            json.dump(cfg, json_file, indent=2)


    def _update_training_stats(self, env_steps, evals):
        self.total_env_steps += env_steps
        self.total_evals += evals

    def report_evals(self):
        now = time.time()
        self.eval_stats.append((now, self.total_env_steps, self.total_evals))
        if len(self.eval_stats) <= 1: return 0.0, 0.0

        fps, eps = [], []
        for avg_interval in self.avg_stats_intervals:
            past_time, past_frames, past_evals = self.eval_stats[max(0, len(self.eval_stats) - 1 - avg_interval)]
            fps.append(round((self.total_env_steps - past_frames) / (now - past_time), 1))
            eps.append(round((self.total_evals - past_evals) / (now - past_time), 1))

        log.debug(f'Evals/sec (EPS) 10 sec: {eps[0]}, 60 sec: {eps[1]}, 300 sec: {eps[2]}, '
                  f'FPS 10 sec: {fps[0]}, 60 sec: {fps[1]}, 300 sec: {fps[2]}')

        self._last_report = now
        return fps[2], eps[2]

    def _maybe_stop_training(self):
        if self.total_evals >= self.max_evals:
            cm.save_archive(self.archive, self.all_actors, self.total_evals, self.filename, self.cfg.save_path)
            self._save_cfg(self.cfg, self.cfg.save_path)
            self.stopped = True
            self.stop.emit()

    def init_loops(self, variation_op, evaluator):
        '''
        This is where we connect signals to slots and start various event loops for training
        :param variation_op: Variation worker mutates the policies
        :param evaluator: Evaluates the fitness of mutated policies
        '''
        self.variation_op, self.evaluator = variation_op, evaluator

        # on startup
        self.event_loop.start.connect(self.evaluator.init_env)  # initialize the envs after we spawn the eval's process so that we don't need to pickle the gym
        self.evaluator.init_elites_map.connect(self.variation_op.init_map)  # initialize the map of elites when starting the runner
        self.variation_op.to_evaluate.connect(self.evaluator.on_evaluate)  # mutating policies kickstarts the evaluator
        self.evaluator.request_new_batch.connect(self.variation_op.evolve_batch)  # evaluator requests more mutated policies when its finished evaluating the current batch
        self.evaluator.request_from_init_map.connect(self.variation_op.init_map)
        # auxiliary connections for logging
        self.evaluator.eval_results.connect(self.on_eval_results)
        self.evaluator.eval_results.connect(self.variation_op.on_eval_results)

        # stop everything when training completes
        self.stop.connect(self.variation_op.on_stop)  # runner stops the variation worker
        self.variation_op.stop.connect(self.evaluator.on_stop)  # variation worker stops the evaluator
