import shutil
import time
import os
import json
import numpy as np
import wandb

from collections import deque
from utils.signal_slot import EventLoop, EventLoopObject, EventLoopProcess, signal, Timer
from utils.logger import log
from utils.utils import cfg_dict, get_checkpoints
from map_elites import common as cm
from map_elites.trainer import Trainer
from typing import List


class Runner(EventLoopObject):
    def __init__(self, cfg, archive, elites_map, all_actors, actors_file, filename, unique_name='runner loop'):
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
        self.all_actors = all_actors
        self.checkpoints_dir = self.cfg.checkpoint_dir
        self.actors_file = actors_file
        self.filename = filename

        # hyperparams
        self.max_evals = cfg.max_evals

        # other event loops
        self.trainers = None
        self.variation_op, self.evaluators = None, None
        self.component_ids = []  # Other EventLoopObject component ids

        # logging variables
        self.avg_stats_intervals = (2, 12, 60)  # 10 seconds, 1 minute, 5 minutes
        self._report_interval = 5.0  # report every 5 seconds
        self._last_report = 0.0
        self.eval_stats = deque([], maxlen=max(self.avg_stats_intervals))
        self.total_env_steps, self.total_evals = 0, 0

        self.periodic(self._report_interval, self.report_evals)
        self.periodic(self.cfg.cp_save_period_sec, self.save_checkpoint)

    @signal
    def stop(self): pass

    @signal
    def release_keys(self): pass  # release keys to policies that have finished being mapped so that the var worker can use them again

    @signal
    def run_iteration(self): pass

    @signal
    def send_keys(self): pass

    @property
    def report_interval(self):
        return self._report_interval

    @property
    def last_report(self):
        return self._last_report

    # timers for periodic logging / saving results
    def periodic(self, period, callback):
        return Timer(self.event_loop, period).timeout.connect(callback)

    def on_evaluator_ready(self):
        self.periodic(3.0, self._maybe_train)  # every 3 seconds, try to run another iteration of each Trainer

    def on_release(self, mutated_actor_keys):
        '''
        The evaluator may need to exit early and release its held keys. It will send the keys here and the runner
        will return the keys to all the mutators
        '''
        self.release_keys.emit(mutated_actor_keys)

    def on_eval_results(self, oid, metadata, evals, frames, runtime, avg_ep_length):
        self._update_training_stats(frames, evals)
        if self.cfg.use_wandb:
            wandb.log({'eval runtime': runtime, 'avg_ep_length': avg_ep_length})
        fit_list = []
        for agent in metadata:
            fit_list.append(agent.fitness)
            self.archive.append(agent)  # add Individual to archive of individuals
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
        self._log_metadata()
        self._maybe_stop_training()

    def _log_metadata(self):
        elites = self.elites_map.values()
        fit_list = [x[1] for x in elites]
        # write log
        log.info(f'n_evals: {self.total_evals}, mean fitness: {np.mean(fit_list)}, median fitness: {np.median(fit_list)}, \
            5th percentile: {np.percentile(fit_list, 5)}, 95th percentile: {np.percentile(fit_list, 95)}')
        fit_list = np.array(fit_list)
        log.info(f'Elites map now contains {len(self.elites_map)} solutions')
        if self.cfg['use_wandb']:
            wandb.log({
                "evals": self.total_evals,
                "mean fitness": np.mean(fit_list),
                "median fitness": np.median(fit_list),
                "5th percentile": np.percentile(fit_list, 5),
                "95th percentile": np.percentile(fit_list, 95),
                "env steps": self.total_env_steps,
                "archive_size": len(self.elites_map)
            })

    def save_checkpoint(self):
        log.debug(f'Saving checkpoint...')
        checkpoint_name = f'checkpoint_{self.total_evals:09d}/'
        filepath = os.path.join(self.checkpoints_dir, checkpoint_name)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self._save_archive(self.archive, self.all_actors, self.total_evals, self.filename, save_path=filepath, save_models=True)
        self._save_cfg(self.cfg, filepath)

        # delete old checkpoints
        while len(get_checkpoints(self.cfg.checkpoint_dir)) > self.cfg.keep_checkpoints:
            oldest_checkpoint = get_checkpoints(self.cfg.checkpoint_dir)[0]
            if os.path.exists(oldest_checkpoint):
                log.debug('Removing %s', oldest_checkpoint)
                shutil.rmtree(oldest_checkpoint)

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

        if self.cfg.use_wandb:
            wandb.log({
                'fps': fps[2],
                'eps': eps[2]
            })

        self._last_report = now
        return fps[2], eps[2]

    def _maybe_train(self):
        self.run_iteration.emit()

    def _maybe_stop_training(self):
        if self.total_evals >= self.max_evals:
            log.debug('Finished training. Saving the archive...')
            cm.save_archive(self.archive, self.all_actors, self.total_evals, self.filename, self.cfg.save_path)
            self._save_cfg(self.cfg, self.cfg.save_path)
            self.stopped = True
            self.stop.emit()

    def _on_component_stopped(self, oid):
        log.debug(f'Stopping component {oid=}')
        for i, id in enumerate(self.component_ids):
            if id == oid:
                del self.component_ids[i]

        if not self.component_ids:
            self.event_loop.stop()

    def init_loops(self, trainers):
        '''
        This is where we connect signals to slots and start various event loops for training
        '''
        self.trainers: List[Trainer:EventLoopObject] = trainers

        for trainer in self.trainers:
            mutator, evaluator = trainer.get_components()

            # run an iteration of training from the Trainer
            self.run_iteration.connect(trainer.train)
            # start the evaluators
            self.event_loop.start.connect(trainer.on_start)
            # evaluator lets runner know it's ready to begin
            trainer.init_success.connect(self.on_evaluator_ready)
            # auxiliary connections for logging
            trainer.eval_results.connect(self.on_eval_results)
            trainer.release_keys.connect(self.on_release)
            # allows all trainers to update their local copies of available keys
            self.release_keys.connect(trainer.on_release_keys)

            # stop the components
            self.stop.connect(trainer.on_stop)
            trainer.stop.connect(self._on_component_stopped)

        self.component_ids = [trainer.object_id for trainer in self.trainers]


    # def init_loops(self, variation_op, evaluators):
    #     '''
    #     This is where we connect signals to slots and start various event loops for training
    #     :param variation_op: Variation worker mutates the policies
    #     :param evaluator: Evaluates the fitness of mutated policies
    #     '''
    #     self.variation_op: VariationOperator = variation_op
    #     self.evaluators: List[Evaluator] = evaluators
    #
    #     # on startup
    #     # self.event_loop.start.connect(self.evaluator.init_env)  # initialize the envs after we spawn the eval's process so that we don't need to pickle the gym
    #     # self.evaluator.init_elites_map.connect(self.variation_op.init_map)  # initialize the map of elites when starting the runner
    #     # self.variation_op.to_evaluate.connect(self.evaluator.on_evaluate)  # mutating policies kickstarts the evaluator
    #
    #     # when runner finishes mapping policies to the archive, hand them back to the var worker
    #     self.release_keys.connect(self.variation_op.on_release)
    #
    #     for evaluator in self.evaluators:
    #         # start the evaluators
    #         self.event_loop.start.connect(evaluator.init_env)
    #         evaluator.init_elites_map.connect(self.variation_op.evolve_batch)
    #         self.variation_op.to_evaluate.connect(evaluator.on_evaluate)
    #
    #         # auxiliary connections for logging
    #         evaluator.eval_results.connect(self.on_eval_results)
    #         evaluator.eval_results.connect(self.variation_op.on_eval_results)
    #
    #         # allow for dynamic resizing of num_envs during runtime
    #         self.variation_op.resize_vec_env.connect(evaluator.resize_env)
    #         # early release keys if mismatch b/w vec-env size and # of mutated agents received
    #         evaluator.release_keys.connect(self.variation_op.on_release)
    #
    #         # stop the evaluators
    #         self.variation_op.stop.connect(evaluator.on_stop)
    #         evaluator.stop.connect(self._on_component_stopped)
    #
    #     # stop everything when training completes
    #     self.stop.connect(self.variation_op.on_stop)  # runner stops the variation worker
    #     # self.variation_op.stop.connect(self.evaluator.on_stop)  # variation worker stops the evaluator
    #
    #     self.component_ids = [variation_op.object_id] + [eval.object_id for eval in self.evaluators]
    #     self.variation_op.stop.connect(self._on_component_stopped)
    #     # self.evaluator.stop.connect(self._on_component_stopped)

    def run(self):
        try:
            self.event_loop.exec()
            self.stop.emit()
        except KeyboardInterrupt:
            log.debug(f'Detected keyboard interrupt in {self.object_id}')
            self.stop.emit()
