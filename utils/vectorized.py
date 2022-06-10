import torch
import torch.nn as nn
import cloudpickle
import pickle
import numpy as np

from torch import Tensor
from models.policy import Policy
from utils.logger import log
from typing import List
from functools import partial

class BatchLinearBlock(nn.Module):
    def __init__(self, weights: Tensor, biases=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = nn.Parameter(weights)  # one slice of all the mlps we want to process as a batch
        self.bias = nn.Parameter(biases)

    def forward(self, x:Tensor) -> Tensor:
        obs_per_weight = x.shape[0] // self.weight.shape[0]
        x = torch.reshape(x, (-1, obs_per_weight, x.shape[1]))
        w_t = torch.transpose(self.weight, 1, 2)
        y = torch.bmm(x, w_t)
        if self.bias is not None:
            y = torch.transpose(y, 0, 1)
            y += self.bias

        out_features = self.weight.shape[1]
        y = torch.reshape(y, shape=(-1, out_features))
        return y


class BatchMLP(Policy):
    def __init__(self, device, model_fn, mlps=None, blocks=None, std_devs=None, num_mlps=0, mlp_ids=None):
        '''
        There are two ways to create a BatchMLP model:
        Method 1: pass in a list of mlps
        Method 2: Pass in a list of layers, std_devs, num_mlps, and mlp_ids. This approach implies that previously,
        two or more BatchMLPs were combined with the combine() method and is manually passing in the mlp metadata from
        the lists of mlps that were used to create those smaller BatchMLP objects
        :param device: device
        :param mlps: list of mlps
        :param layers: list of layers from smaller, equally sized BatchMLP objects
        :param std_devs: std_devs from the BatchMLP objects
        :param num_mlps: metadata info from the BatchMLP objects
        :param mlp_ids: metadata info from the BatchMLP objects
        '''
        super().__init__()
        self.device = device
        self.model_fn: partial[nn.Module] = model_fn

        if blocks is not None:
            assert std_devs is not None, log.error(f'Must pass std_dev parameters from other BatchMLPs to create a new BatchMLP')
            self.num_mlps = num_mlps
            self.mlp_ids = mlp_ids
            self.layers = nn.Sequential(*blocks)
            self._action_log_std = nn.Parameter(torch.cat(std_devs)).to(device)
        else:
            assert mlps is not None, log.error(f'Attempted to build the BatchMLP object using an ndarray of mlps, but no mlps were passed')
            assert isinstance(mlps, np.ndarray), 'mlps should be passed as a numpy ndarray'
            self.num_mlps = len(mlps)
            self.mlp_ids = [mlp.id for mlp in mlps]
            self.blocks = self._slice_mlps(mlps)
            self.layers = nn.Sequential(*self.blocks)

            self._action_log_std = []
            for mlp in mlps:
                self._action_log_std.append(mlp.action_log_std.detach().clone())
            self._action_log_std = nn.Parameter(torch.vstack(self._action_log_std)).to(device)

    def forward(self, x):
        return self.layers(x)

    @property
    def action_log_std(self):
        return self._action_log_std

    def set_parent_id(self, which_parent, ids):
        assert which_parent == 1 or which_parent == 2, 'invalid parent value. Can only have 2 parents (parent 1 or parent 2)'
        for mlp, id in zip(self.mlps, ids):
            if which_parent == 1:
                mlp.parent_1_id = id
            else:
                mlp.parent_2_id = id

    def get_mlp_ids(self):
        return self.mlp_ids

    def _slice_mlps(self, mlps):
        num_layers = len(mlps[0].layers)
        blocks = []
        # slice_weights = torch.tensor([self.mlps[i].layers[j] for j in range(num_layers) for i in range(len(self.mlps)) if isinstance(self.mlps[i].layers[j], nn.Linear)])
        for i in range(0, num_layers):
            if not isinstance(mlps[0].layers[i], nn.Linear):
                continue
            slice_weights = [mlps[j].layers[i].weight.to(self.device) for j in range(self.num_mlps)]
            slice_bias = [mlps[j].layers[i].bias.to(self.device) for j in range(self.num_mlps)]
            slice_weights = torch.stack(slice_weights)
            slice_bias = torch.stack(slice_bias)
            nonlinear = mlps[0].layers[i+1] if i+1 < num_layers else None
            block = BatchLinearBlock(slice_weights, slice_bias)
            blocks.append(block)
            if nonlinear is not None:
                blocks.append(nonlinear)
        return blocks

    def _get_mlp_at_idx(self, idx):
        # create the mlp itself
        mlp = self.model_fn(self.device)
        # update the weights
        for l, layer in enumerate(self.layers):
            if not isinstance(layer, BatchLinearBlock):
                continue
            mlp.layers[l].weight.data = layer.weight.data[idx]
            mlp.layers[l].bias.data = layer.bias.data[idx]
        # update the mlp id
        mlp.id = self.mlp_ids[idx]
        # update the mlp log-stddev
        mlp.action_log_std.data = self.action_log_std[idx].data
        return mlp

    def _update_mlp_at_idx(self, idx, mlp):
        # update the weights
        for l, layer in enumerate(self.layers):
            if not isinstance(layer, BatchLinearBlock):
                continue
            layer.weight.data[idx] = mlp.layers[l].weight.data
            layer.bias.data[idx] = mlp.layers[l].bias.data
        # update the mlp id
        self.mlp_ids[idx] = mlp.id
        # update the log_stddev
        self.action_log_std[idx].data = mlp.action_log_std.detach().clone()

    # def update_mlps(self, mlps: List[nn.Module]):
    #     '''
    #     Update the list of mlps with the whatever parameters are stored in the BatchMLP object
    #     The mlps that produced this object and the mlps argument must be the same size/architecture
    #     :param mlps: list of mlps to update
    #     :return: list of mlps with the new parameters
    #     '''
    #     assert len(mlps) == self.num_mlps, log.error(f'List of mlps to update must match the length of mlps that was \
    #                                                     used to build this BatchMLP. {self.num_mlps=}, but received {len(mlps)} mlps')
    #     for i, layer in enumerate(self.layers):
    #         for j in range(len(mlps)):
    #             if not isinstance(mlps[j].layers[i], nn.Linear):
    #                 continue
    #             mlps[j].layers[i].weight.data = layer.weight[j]
    #             mlps[j].layers[i].bias.data = layer.bias[j]
    #     # update the stddev tensors
    #     log_stddevs = self.action_log_std.view(len(mlps), -1)
    #     for log_stddev, mlp in zip(log_stddevs, mlps):
    #         mlp.action_log_std.data = log_stddev
    #     return mlps

    # def update_batch_mlps(self, batch_mlps):
    #     '''
    #     Similar to update_mlps(), but instead we update the weights of each BatchMLP object in a list of BatchMLPs
    #     :param batch_mlps:
    #     :return:
    #     '''
    #     num_batch_mlps = self.num_mlps // batch_mlps[0].num_mlps
    #     assert num_batch_mlps == len(batch_mlps), log.error(f'Number of batch_mlps passed in does not match the size of this BatchMLP')
    #
    #     for i, layer in enumerate(self.layers):
    #         if not isinstance(layer, BatchLinearBlock):
    #             continue
    #         # need to reshape weights and biases since there are {mutations_per_policy} mlps per batch_mlp
    #         layer.weight.data = layer.weight.view(num_batch_mlps, batch_mlps[0].num_mlps, *layer.weight.shape[1:])
    #         layer.bias.data = layer.bias.view(num_batch_mlps, batch_mlps[0].num_mlps, *layer.bias.shape[1:])
    #         for j in range(len(batch_mlps)):
    #             batch_mlps[j].layers[i].weight.data = layer.weight[j]
    #             batch_mlps[j].layers[i].bias.data = layer.bias[j]
    #     # update the stddev tensors
    #     log_stddevs = self.action_log_std.view(num_batch_mlps, -1)
    #     for log_stddev, mlp in zip(log_stddevs, batch_mlps):
    #         mlp.action_log_std.data = log_stddev
    #     return batch_mlps

    # def update_self(self, mlps):
    #     '''
    #     Opposite operation of update_mlps(). Use the weights of the mlps to update the weights of the BatchMLP
    #     '''
    #     assert isinstance(mlps, np.ndarray), 'mlps should be passed as a numpy ndarray'
    #
    #     for i, layer in enumerate(self.layers):
    #         for j in range(len(mlps)):
    #             if not isinstance(mlps[j].layers[i], nn.Linear):
    #                 continue
    #             layer.weight[j] = mlps[j].layers[i].weight.data  # TODO: make sure this works
    #             layer.bias[j] = mlps[j].layers[i].bias.data
    #     # update the std-dev tensors
    #     log_std = []
    #     for mlp in mlps:
    #         log_std.append(mlp.action_log_std.detach().clone())
    #     self.action_log_std.data = torch.cat(log_std).to(self.device)  # TODO: make sure this works

    def to_device(self, device):
        self.to(device)
        for i in range(len(self.mlps)):
            self.mlps[i].to(device)

    def __len__(self):
        return self.num_mlps

    # override getter and setter for BatchMLP slicing
    def __getitem__(self, key):  # TODO: test
        if isinstance(key, slice):
            inds = range(*key.indices(self.num_mlps))
            mlps = []
            for ind in inds:
                mlps.append(self._get_mlp_at_idx(ind))
            return BatchMLP(self.device, self.model_fn, np.array(mlps))
        if isinstance(key, tuple):
            mlps = []
            for idx in key:
                mlps.append(self._get_mlp_at_idx(idx))
            return BatchMLP(self.device, self.model_fn, np.array(mlps))
        return self._get_mlp_at_idx(key)

    def __setitem__(self, key, value):  # TODO: test
        if isinstance(key, slice):
            inds = range(*key.indices(self.num_mlps))
            for idx in inds:
                self._update_mlp_at_idx(idx, value[idx])
        elif isinstance(key, tuple):
            for idx, mlp in zip(key, value):
                self._update_mlp_at_idx(idx, mlp)
        else:
            self._update_mlp_at_idx(key, value)


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


def combine(batch_mlps: List[BatchMLP]):
    '''
    Combine smaller batchMLPs into one larger BatchMLP
    :param batch_mlps: list of smaller BatchMlps
    :return: BatchMLP
    '''
    all_layers, std_devs, mlp_ids = [], [], []
    device = batch_mlps[0].device
    num_mlps = 0
    for b in batch_mlps:
        all_layers.append(b.layers)
        std_devs.append(b.action_log_std)
        num_mlps += b.num_mlps
        mlp_ids.extend(b.get_mlp_ids())

    num_layers_one_mlp = len(all_layers[0])
    blocks = []
    for i in range(num_layers_one_mlp):
        if isinstance(all_layers[0][i], BatchLinearBlock):
            weights = torch.cat([all_layers[l][i].weight for l in range(len(all_layers))], dim=0)
            biases = torch.cat([all_layers[l][i].bias for l in range(len(all_layers))], dim=0)
            new_block = BatchLinearBlock(weights, biases)
            blocks.append(new_block)
        else:  # nonlinearity
            blocks.append(all_layers[0][i])
    res = BatchMLP(device, mlps=None, blocks=blocks, std_devs=std_devs, num_mlps=num_mlps, mlp_ids=mlp_ids)
    return res