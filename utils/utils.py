import os
import glob
import json
import numpy as np
from map_elites import common as cm
from pynvml import *

def get_checkpoints(checkpoints_dir):
    checkpoints = glob.glob(os.path.join(checkpoints_dir, 'checkpoint_*'))
    return sorted(checkpoints)


def save_checkpoint(archive, n_evals, archive_name, checkpoints_dir, cfg):
    checkpoint_name = f'checkpoint_{n_evals:09d}/'
    filepath = os.path.join(checkpoints_dir, checkpoint_name)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    cm.save_archive(archive, n_evals, archive_name, save_path=filepath, save_models=True)
    save_cfg(cfg, filepath)


def cfg_dict(cfg):
    if isinstance(cfg, dict):
        return cfg
    else:
        return vars(cfg)


def     save_cfg(cfg, save_path):
    cfg = cfg_dict(cfg)
    cfg_file = os.path.join(save_path, 'cfg.json')
    with open(cfg_file, 'w') as json_file:
        json.dump(cfg, json_file, indent=2)


def get_least_busy_gpu(num_gpus):
    gpu_ids = [i for i in range(num_gpus)]
    free_mem = []
    for gpu_id in gpu_ids:
        h = nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvmlDeviceGetMemoryInfo(h)
        free_mem.append(info.free)
    return np.argmax(free_mem)