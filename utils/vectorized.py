import torch
import torch.nn as nn
import cloudpickle
import pickle

from torch import Tensor




class BatchLinearBlock(nn.Module):
    def __init__(self, weights: Tensor, nonlinear, biases=None, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = nn.Parameter(weights)  # one slice of all the mlps we want to process as a batch
        self.bias = nn.Parameter(biases)
        self.nonlinear = nonlinear

    def forward(self, x:Tensor) -> Tensor:
        obs_per_weight = x.shape[0] // self.weight.shape[0]
        x = torch.reshape(x, (-1, obs_per_weight, x.shape[1]))
        w_t = torch.transpose(self.weight, 1, 2)
        y = torch.bmm(x, w_t)
        if self.bias is not None:
            y = torch.transpose(y, 0, 1)
            y += self.bias
        if self.nonlinear is not None:
            y = self.nonlinear(y)

        out_features = self.weight.shape[1]
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

    def set_parent_id(self, which_parent, ids):
        assert which_parent == 1 or which_parent == 2, 'invalid parent value. Can only have 2 parents (parent 1 or parent 2)'
        for mlp, id in zip(self.mlps, ids):
            if which_parent == 1:
                mlp.parent_1_id = id
            else:
                mlp.parent_2_id = id

    def get_mlp_ids(self):
        ids = [mlp.id for mlp in self.mlps]
        return ids

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

    def update_mlps(self):
        '''
        The batch mlp blocks may have been updated from backprop or mutation/crossover.
        This method updates the weights of the original mlps that produced the batch blocks
        '''
        for i, layer in enumerate(self.layers):
            for j in range(len(self.mlps)):
                # 2 * i b/c every other layer is an activation
                self.mlps[j].layers[2 * i].weight.data = layer.weight[i]
                self.mlps[j].layers[2 * i].bias.data = layer.bias[i]
        return self.mlps

    def to_device(self, device):
        self.to(device)
        for i in range(len(self.mlps)):
            self.mlps[i].to(device)


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