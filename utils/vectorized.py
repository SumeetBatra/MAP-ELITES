import torch
import time
import cloudpickle
import pickle
import numpy as np
from faster_fifo import Queue
from multiprocessing import Process, Event, Pipe
from utils.logger import log


# adapted from: https://github.com/ollenilsson19/PGA-MAP-Elites/blob/master/vectorized_env.py


def parallel_worker(process_id,
                    env_fn_wrapper,
                    eval_in_queue,
                    eval_out_queue,
                    trans_out_queue,
                    close_processes,
                    remote,
                    master_seed):
    '''
    Function that runs the paralell processes for the evaluation
    Parameters:
        process_id (int): ID of the process so it can be identified
        env_fn_wrapper : function that when called starts a new environment
        eval_in_queue (Queue object): queue for incoming actors
        eval_out_queue (Queue object): queue for outgoing actors
        trans_out_queue (Queue object): queue for outgoing transitions
    '''
    # start env simulation
    env = env_fn_wrapper.x()
    # begin process loop
    while True:
        try:
            # get a new actor to evaluate
            try:
                idx, actor, eval_id, eval_mode = eval_in_queue.get_nowait()
                env.seed(int((master_seed + 100) * eval_id))
                obs = env.reset()
                done = False
                # eval loop
                obs_arr, rew_arr, dones_arr = [], [], []
                rewards, info = 0, None
                while not done:
                    with torch.no_grad():
                        obs = torch.from_numpy(obs).to(actor.device)
                        action = actor(obs).cpu().detach().numpy()
                        obs, rew, done, info = env.step(action)
                        obs_arr.append(obs)
                        rew_arr.append(rew)
                        dones_arr.append(done)
                        rewards += rew
                eval_out_queue.put((idx, (rewards, env.ep_length, info['desc'])))
            except BaseException:
                pass
            if close_processes.is_set():
                log.debug(f'Close Eval Process id {process_id}')
                remote.send(process_id)  # TODO: add rng state??
                env.close()
                time.sleep(5)
                break

        except KeyboardInterrupt:
            env.close()
            break


class ParallelEnv(object):
    def __init__(self, env_fns, batch_size, random_init, seed):
        """
        A class for parallel evaluation
        """
        self.n_processes = len(env_fns)
        self.eval_in_queue = Queue(max_size_bytes=int(1e7))
        self.eval_out_queue = Queue(max_size_bytes=int(1e7))
        self.trans_out_queue = Queue()
        self.remotes, self.locals = zip(*[Pipe() for _ in range(self.n_processes + 1)])
        self.global_sync = Event()
        self.close_processes = Event()

        self.steps = None
        self.batch_size = batch_size
        self.seed = seed
        self.eval_id = 0

        self.processes = [Process(target=parallel_worker,
                                  args=(process_id,
                                        CloudpickleWrapper(env_fn),
                                        self.eval_in_queue,
                                        self.eval_out_queue,
                                        self.trans_out_queue,
                                        self.close_processes,
                                        self.remotes[process_id],
                                        self.seed)) for process_id, env_fn in enumerate(env_fns)]

        for p in self.processes:
            p.daemon = True
            p.start()

    def eval_policy(self, actors, eval_mode=False):
        self.steps = 0
        results = [None] * len(actors)
        for idx, actor in enumerate(actors):
            self.eval_id += 1
            self.eval_in_queue.put((idx, actor, self.eval_id, eval_mode), block=True, timeout=1e9)  # faster-fifo queue is 10s timeout by default
        # EXPERIMENTAL CODE
        # res = self.eval_out_queue.get_many(True, timeout=1e9)
        # inds, res = list(zip(*res))
        # self.steps = sum(r[1] for r in res)
        # results[list(inds)] = res

        for _ in range(len(actors)):
            idx, res = self.eval_out_queue.get(True, timeout=1e9)  # faster-fifo queue is 10s timeout by default
            self.steps += res[1]
            results[idx] = res
        return results

    def update_archive(self, archive):
        self.locals[-1].send(archive)

    def get_actors(self):
        pass

    def close(self):
        self.close_processes.set()
        rng_states = []
        for local in self.locals[0:-1]:
            rng_states.append(local.recv())  # TODO: what does this do?
        for p in self.processes:
            p.terminate()

        # TODO: get rid of this?
        # return [[x[1] for x in sorted(rng_states, key=lambda element: element[0])]]


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_env.py#L190
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)