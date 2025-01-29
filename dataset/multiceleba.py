"""
Author: Rui Hu
All rights reserved.
"""

import numpy as np
import json
from pathlib import Path

import random
import torch
from torch.utils.data import Dataset
import torchvision.datasets as tv_dataset


class BiasedCelebA(Dataset):
    raw_celeba = tv_dataset.CelebA('/vol/research/project_storage/data/celebA/', 'all')

    obj_name_list = [
        "smiling",
        "sad",
    ]

    bg_name_list = [
        "smiling ",
        "sad",
    ]

    co_occur_obj_name_list = [
        "smiling",
        "sad",
    ]

    def __init__(self, root, target_name, biasA_name, biasB_name, biasA_ratio, biasB_ratio, split, transform=None):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.transform = transform
        self.return_group_index = False

        base_folder = f"/vol/research/project_storage/data/celeba-MB/{target_name}-{biasA_name}-{biasA_ratio}-{biasB_name}-{biasB_ratio}"
        data_dir = Path(root) / base_folder / f"{split}_data.npy"

        self.data = np.load(open(data_dir, 'rb'))
        self.target = self.data[:, 1:]
        self.biasA_ratio = biasA_ratio
        self.biasB_ratio = biasB_ratio
        self.obj_label = self.target[:, 0]
        self.return_contrastive_pairs = False

        self.group_num = 8
        self.meta_key = f"{target_name}__{biasA_name}__{biasB_name}"
        self.group_label, self.biasA_is_aligned, self.biasB_is_aligned = self._setup_group_label_is_aligned()

    def __getitem__(self, index):
        raw_index = self.data[index][0]
        target = self.data[index][1:]
        image = self.raw_celeba[raw_index][0]


        if self.transform:
            image = self.transform(image)

        target = torch.as_tensor(target)
        data_dict = {"image": image, "label": target, "index": index}

        if self.return_contrastive_pairs:
            rank = self.ranks[index]
            max_rank = int(self.max_rank[self.obj_label[index].item()])
            pos_rank = int(min(rank + self.gap, max_rank))
            data_dict['pos_labels'] = self.obj_label[index]
            neg_idx = np.random.choice(self.indices_label_rank[self.obj_label[index].item()][rank])
            try:
                pos_idx = np.random.choice(self.indices_label_rank[self.obj_label[index].item()][pos_rank])
            except (KeyError, ValueError):
                found = False
                for fallback_rank in range(pos_rank + 1, max_rank + 1):
                    indices = self.indices_label_rank[self.obj_label[index].item()].get(fallback_rank, [])
                    if len(indices) > 0:
                        pos_idx = np.random.choice(indices)
                        found = True
                        break
                if not found:
                    pos_idx = -1
            if pos_idx != -1:
                raw_index = self.data[pos_idx][0]
                image = self.raw_celeba[raw_index][0]
                data_dict['positive'] = self.transform(image)
            else:
                return None
            raw_index = self.data[neg_idx][0]
            image = self.raw_celeba[raw_index][0]
            data_dict['negative'] = self.transform(image)
        return data_dict

    def __len__(self):
        return len(self.data)

    def _get_subsample_group_indices(self, subsample_which_shortcut):
        bg_ratio = self.biasA_ratio
        co_occur_obj_ratio = self.biasB_ratio

        num_img_per_obj_class = len(self) // len(self.obj_name_list)
        if subsample_which_shortcut == "bg":
            min_size = int(min(1 - bg_ratio, bg_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "co_occur_obj":
            min_size = int(min(1 - co_occur_obj_ratio, co_occur_obj_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "both":
            min_bg_ratio = min(1 - bg_ratio, bg_ratio)
            min_co_occur_obj_ratio = min(1 - co_occur_obj_ratio, co_occur_obj_ratio)
            min_size = int(min_bg_ratio * min_co_occur_obj_ratio * num_img_per_obj_class)
        else:
            raise NotImplementedError

        assert min_size > 1

        indices = []

        if subsample_which_shortcut == "both":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.target[:, 0] == idx_obj
                for idx_bg in range(len(self.bg_name_list)):
                    bg_mask = self.target[:, 1] == idx_bg
                    for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                        co_occur_obj_mask = self.target[:, 2] == idx_co_occur_obj
                        mask = obj_mask & bg_mask & co_occur_obj_mask
                        mask = torch.tensor(mask, dtype=torch.bool)
                        subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                        random.shuffle(subgroup_indices)
                        sampled_subgroup_indices = subgroup_indices[:min_size]
                        indices += sampled_subgroup_indices
        else:
            raise NotImplementedError

        return indices

    def _setup_group_label_is_aligned(self):
        metadata = json.load(open("create_datasets/celeba/metadata.json", 'r'))
        assert self.meta_key in metadata.keys()

        group_def = list(metadata[self.meta_key].values())
        group_list = [group_def[0][0], group_def[1][0], group_def[2][0], group_def[3][0]] + \
                     [group_def[0][1], group_def[1][1], group_def[2][1], group_def[3][1]]

        group_label = []
        for _target in self.target:
            for group_id, group_info in enumerate(group_list):
                if (
                        (_target[0] == group_info['target']) &
                        (_target[1] == group_info['biasA']) &
                        (_target[2] == group_info['biasB'])
                ):
                    group_label.append(group_id)
                    break
        group_label = torch.as_tensor(
            group_label,
            dtype=torch.long
        )

        biasA_is_aligned = torch.as_tensor(
            (group_label == 0) | (group_label == 1) | (group_label == 4) | (group_label == 5),
            dtype=torch.long
        )
        biasB_is_aligned = torch.as_tensor(
            (group_label == 0) | (group_label == 2) | (group_label == 4) | (group_label == 6),
            dtype=torch.long
        )

        return group_label, biasA_is_aligned, biasB_is_aligned

    def set_num_group_and_group_array(self, num_shortcut_category, shortcut_label):
        self.num_group = len(self.obj_name_list) * num_shortcut_category
        self.group_array = torch.Tensor(self.obj_label)* num_shortcut_category + shortcut_label

    def get_sampling_weights(self):
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array.long()]
        return weights
