import torch
import torch.nn as nn
import cloudpickle
import pickle
import numpy as np

from torch import Tensor
from models.policy import Policy


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
    def __init__(self, mlps, device):
        super().__init__()
        assert isinstance(mlps, np.ndarray), 'mlps should be passed as a numpy ndarray'
        self.num_mlps = len(mlps)
        self.mlp_ids = [mlp.id for mlp in mlps]
        self.device = device
        self.blocks = self._slice_mlps(mlps)
        self.layers = nn.Sequential(*self.blocks)
        self._action_log_std = []

        for mlp in mlps:
            self._action_log_std.append(mlp.action_log_std.detach().clone())
        self._action_log_std = nn.Parameter(torch.cat(self._action_log_std)).to(device)

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

    # def replace_mlps(self, ids, mlps):
    #     self.mlps[ids] = mlps
    #     for name, weight in self.named_parameters():
    #         weight.data[ids] = torch.stack([mlp.state_dict()[name] for mlp in mlps]).to(self.device)

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

    # def update_mlps(self):
    #     '''
    #     The batch mlp blocks may have been updated from backprop or mutation/crossover.
    #     This method updates the weights of the original mlps that produced the batch blocks
    #     '''
    #     for i, layer in enumerate(self.layers):
    #         for j in range(len(self.mlps)):
    #             if not isinstance(self.mlps[j].layers[i], nn.Linear):
    #                 continue
    #             self.mlps[j].layers[i].weight.data = layer.weight[j]
    #             self.mlps[j].layers[i].bias.data = layer.bias[j]
    #     return self.mlps

    def update_mlps(self, mlps):
        '''
        Update the list of mlps with the whatever parameters are stored in the BatchMLP object
        The mlps that produced this object and the mlps argument must be the same size/architecture
        :param mlps: list of mlps to update
        :return: list of mlps with the new parameters
        '''
        for i, layer in enumerate(self.layers):
            for j in range(len(mlps)):
                if not isinstance(mlps[j].layers[i], nn.Linear):
                    continue
                mlps[j].layers[i].weight.data = layer.weight[j]
                mlps[j].layers[i].bias.data = layer.bias[j]
        # update the stddev tensors
        log_stddevs = self.action_log_std.view(len(mlps), -1)
        for log_stddev, mlp in zip(log_stddevs, mlps):
            mlp.action_log_std.data = log_stddev
        return mlps


    def to_device(self, device):
        self.to(device)
        for i in range(len(self.mlps)):
            self.mlps[i].to(device)

    def __len__(self):
        return self.num_mlps


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