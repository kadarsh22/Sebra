"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import random
import torch
import numpy as np
import argparse
import copy
import os
from typing import List
import glob
from PIL import Image
import torch.nn as nn

from torch.utils.data.dataset import Dataset
from torchvision import transforms

EPS = 1e-6


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def lerp(lam, t1, t2):
    t3 = copy.deepcopy(t2)
    for p in t1:
        t3[p] = (1 - lam) * t1[p] + lam * t2[p]
    return t3


class MultiDimAverageMeter:
    # reference: https://github.com/alinlab/LfF/blob/master/util.py

    def __init__(self, dims):
        self.dims = dims
        self.eye_tsr = torch.eye(dims[0]).long()
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )

    def get_worst_group_acc(self):
        num_correct = self.cum.reshape(*self.dims)
        cnt = self.cnt.reshape(*self.dims)

        first_shortcut_worst_group_acc = (
                num_correct.sum(dim=2) / cnt.sum(dim=2)
        ).min()
        second_shortcut_worst_group_acc = (
                num_correct.sum(dim=1) / cnt.sum(dim=1)
        ).min()
        both_worst_group_acc = (num_correct / cnt).min()

        return (
            first_shortcut_worst_group_acc,
            second_shortcut_worst_group_acc,
            both_worst_group_acc,
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name} {avg:.3f}"
        return fmtstr.format(**self.__dict__)


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return idx, self.dataset[idx]


class EMAGPU:
    def __init__(self, label, device, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.device = device
        self.parameter = torch.zeros(label.size(0), device=device)
        self.updated = torch.zeros(label.size(0), device=device)
        self.num_class = label.max().item() + 1
        self.max_param_per_class = torch.zeros(self.num_class, device=device)

    def update(self, data, index):
        self.parameter[index] = (
                self.alpha * self.parameter[index]
                + (1 - self.alpha * self.updated[index]) * data
        )
        self.updated[index] = 1

        # update max_param_per_class
        batch_size = len(index)
        buffer = torch.zeros(batch_size, self.num_class, device=self.device)
        buffer[range(batch_size), self.label[index]] = self.parameter[index]
        cur_max = buffer.max(dim=0).values
        global_max = torch.maximum(cur_max, self.max_param_per_class)
        label_set_indices = self.label[index].unique()
        self.max_param_per_class[label_set_indices] = global_max[
            label_set_indices
        ]

    def max_loss(self, label):
        return self.max_param_per_class[label]


class MultiStepLR():
    def __init__(self, milestones: List[int], gamma=0, q_init=0.7):
        super().__init__()
        self.milestones = [int(x) for x in milestones]
        self.gamma = gamma
        self.epoch = 0
        self.q_init = q_init

    def get_q(self):
        if self.epoch not in self.milestones:
            return self.q_init
        return self.q_init * self.gamma

    def step(self):
        self.epoch = self.epoch + 1


def add_prefix_dict(prefix, dictionary):
    modified_dict = {}
    for k, v in dictionary.items():
        if k != 'custom_step':
            modified_dict[prefix + k] = v
        else:
            modified_dict[k] = v
    return modified_dict


def slurm_wandb_argparser():
    parser = argparse.ArgumentParser(add_help=False)
    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_entity", type=str)

    # SLURM
    parser.add_argument("--slurm_job_name", type=str)
    parser.add_argument("--slurm_constraint", type=str)
    parser.add_argument("--slurm_partition", type=str)
    parser.add_argument("--slurm_mem_gb", type=int, default=128)
    parser.add_argument("--slurm_log_dir", type=str, default="exp/logs")

    return parser


def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] = nn.Parameter(torch.from_numpy(weights[index:index + param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m


def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
        total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params