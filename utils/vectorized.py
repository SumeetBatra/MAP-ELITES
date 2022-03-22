import torch
import torch.nn as nn
import time
import cloudpickle
import pickle
import numpy as np
from faster_fifo import Queue
from multiprocessing import Process, Event, Pipe

from torch import Tensor

from utils.logger import log
from utils.utils import get_least_busy_gpu
from pynvml import *




# # adapted from: https://github.com/ollenilsson19/PGA-MAP-Elites/blob/master/vectorized_env.py
# def parallel_worker(process_id,
#                     env_fn_wrapper,
#                     eval_in_queue,
#                     eval_out_queue,
#                     trans_out_queue,
#                     close_processes,
#                     remote,
#                     master_seed):
#     '''
#     Function that runs the paralell processes for the evaluation
#     Parameters:
#         process_id (int): ID of the process so it can be identified
#         env_fn_wrapper : function that when called starts a new environment
#         eval_in_queue (Queue object): queue for incoming actors
#         eval_out_queue (Queue object): queue for outgoing actors
#         trans_out_queue (Queue object): queue for outgoing transitions
#     '''
#     # start env simulation
#     env = env_fn_wrapper.x()
#     # begin process loop
#     while True:
#         try:
#             # get a new actor to evaluate
#             try:
#                 idx, actor, eval_id, eval_mode = eval_in_queue.get_nowait()
#                 env.seed(int((master_seed + 100) * eval_id))
#                 obs = env.reset()
#                 done = False
#                 # eval loop
#                 obs_arr, rew_arr, dones_arr = [], [], []
#                 rewards, info = 0, None
#                 while not done:
#                     with torch.no_grad():
#                         obs = torch.from_numpy(obs).to(actor.device)
#                         action = actor(obs).cpu().detach().numpy()
#                         obs, rew, done, info = env.step(action)
#                         obs_arr.append(obs)
#                         rew_arr.append(rew)
#                         dones_arr.append(done)
#                         rewards += rew
#                 eval_out_queue.put((idx, (rewards, env.ep_length, info['desc'])))
#             except BaseException:
#                 pass
#             if close_processes.is_set():
#                 log.debug(f'Close Eval Process id {process_id}')
#                 remote.send(process_id)  # TODO: add rng state??
#                 env.close()
#                 time.sleep(5)
#                 break
#
#         except KeyboardInterrupt:
#             env.close()
#             break

def parallel_worker(process_id,
                 env_fn_wrappers,
                 eval_in_queue,
                 eval_out_queue,
                 trans_out_queue,
                 close_processes,
                 remote,
                 master_seed,
                 num_gpus):

    nvmlInit()  # for tracking gpu available resources
    # start the simulations
    envs = [env_fn_wrappers.x[i]() for i in range(len(env_fn_wrappers.x))]
    # begin the process loop
    while True:
        try:
            # get a batch of new actors to evaluate
            try:
                idx, actors, eval_id, eval_mode = eval_in_queue.get_nowait()
                assert len(envs) % len(actors) == 0, 'Number of envs should be a multiple of the number of policies '
                gpu_id = get_least_busy_gpu(num_gpus)
                batch_actors = BatchMLP(actors, device=torch.device(f'cuda:{gpu_id}'))
                num_actors = len(actors)
                for env in envs:
                    env.seed(int((master_seed * 100) * eval_id))
                obs = torch.tensor([env.reset() for env in envs]).reshape(num_actors, -1).to(actors[0].device)
                done = False
                rews = [0 for _ in range(num_actors)]
                dones = [False for _ in range(num_actors)]
                infos = [None for _ in range(num_actors)]
                while not all(dones):
                    obs_arr = []
                    with torch.no_grad():
                        acts = batch_actors(obs).cpu().detach().numpy()
                        for idx, (act, env) in enumerate(zip(acts, envs)):
                            obs, rew, done, info = env.step(act)
                            rews[idx] += rew
                            obs_arr.append(obs)
                            dones[idx] = done
                            infos[idx] = info
                        obs = torch.tensor(obs_arr).reshape(num_actors, -1).to(actors[0].device)
                ep_lengths = [env.ep_length for env in envs]
                bds = [info['desc'] for info in infos]  # list of behavioral descriptors
                res = [[rew, ep_len, bd] for rew, ep_len, bd in zip(rews, ep_lengths, bds)]
                eval_out_queue.put_many(res)
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
    def __init__(self, env_fns, batch_size, seed, num_parallel, actors_per_worker, num_gpus):
        """
        A class for parallel evaluation
        """
        self.n_processes = num_parallel
        self.num_gpus = num_gpus
        self.actors_per_worker = actors_per_worker
        self.eval_in_queue = Queue(max_size_bytes=int(1e8))
        self.eval_out_queue = Queue(max_size_bytes=int(1e8))
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
                                        self.seed,
                                        self.num_gpus)) for process_id, env_fn in enumerate(env_fns)]

        for p in self.processes:
            p.daemon = True
            p.start()

    def batch_eval_policies(self, actors, eval_mode=False):
        self.steps = 0
        batch_actors = np.array(actors).reshape(-1, self.actors_per_worker)
        for idx, batch in enumerate(batch_actors):
            self.eval_in_queue.put((idx, batch.tolist(), self.eval_id, eval_mode), block=True, timeout=1e9)

        actors_batch_size = len(actors)
        results = []
        while len(results) < actors_batch_size:
            try:
                res = self.eval_out_queue.get_many_nowait()
                results += res
            except BaseException:
                pass
        self.steps += sum([res[1] for res in results])
        return results


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


class BatchLinearBlock(nn.Module):
    def __init__(self, weights: Tensor, nonlinear, biases=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weights = weights  # one slice of all the mlps we want to process as a batch
        self.biases = biases
        self.nonlinear = nonlinear

    def forward(self, x:Tensor) -> Tensor:
        obs_per_weight = x.shape[0] // self.weights.shape[0]
        x = torch.reshape(x, (-1, obs_per_weight, x.shape[1]))
        w_t = torch.transpose(self.weights, 1, 2)
        y = torch.bmm(x, w_t)
        if self.biases is not None:
            y = torch.transpose(y, 0, 1)
            y += self.biases
        if self.nonlinear is not None:
            y = self.nonlinear(y)

        out_features = self.weights.shape[1]
        y = torch.reshape(y, shape=(-1, out_features))
        return y


class BatchMLP(nn.Module):
    def __init__(self, mlps, device):
        super().__init__()
        self.mlps = mlps
        self.num_mlps = len(mlps)
        self.device = device
        self.blocks = self._slice_mlps()
        self.layers = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.layers(x)

    def _slice_mlps(self):
        num_layers = len(self.mlps[0].layers)
        blocks = []
        # slice_weights = torch.tensor([self.mlps[i].layers[j] for j in range(num_layers) for i in range(len(self.mlps)) if isinstance(self.mlps[i].layers[j], nn.Linear)])
        for i in range(0, num_layers):
            if not isinstance(self.mlps[0].layers[i], nn.Linear):
                continue
            slice_weights = [self.mlps[j].layers[i].weight.to(self.device) for j in range(self.num_mlps)]
            slice_bias = [self.mlps[j].layers[i].bias.to(self.device) for j in range(self.num_mlps)]
            slice_weights = torch.stack(slice_weights)
            slice_bias = torch.stack(slice_bias)
            nonlinear = self.mlps[0].layers[i+1] if i+1 < num_layers else None
            block = BatchLinearBlock(slice_weights, nonlinear, slice_bias)
            blocks.append(block)
        return blocks


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