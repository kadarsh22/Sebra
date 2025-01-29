
import os
import torch
import json
import pickle
import numpy as np


from typing import Any
from torchvision.datasets.folder import ImageFolder


class ImageNet(ImageFolder):
    base_folder = "imagenet"

    def __init__(
        self,
        root: str,
        split: str = "train",
        return_group_index=False,
        return_file_path=False,
        return_dist_shift_index=False,
        return_image_size=False,
        dist_shift_index=0,
        **kwargs: Any
    ) -> None:
        assert split in ["train", "val"]
        root = self.root = os.path.join(root, self.base_folder)
        self.split = split
        wnid_to_classes = self.load_meta_file(self.root)

        super().__init__(self.split_folder, **kwargs)
        self.root = root
        self.return_group_index = return_group_index
        self.return_file_path = return_file_path
        self.return_dist_shift_index = return_dist_shift_index
        self.return_image_size = return_image_size
        self.dist_shift_index = dist_shift_index

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for idx, clss in enumerate(self.classes) for cls in clss
        }
        self.return_contrastive_pairs = False

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def load_meta_file(self, root):
        fpath = os.path.join(root, "labels.json")
        with open(fpath, 'r') as file:
            # Load the JSON data into a Python dictionary
            data = json.load(file)

        wnid_to_classes = {}

        for _, val in data.items():
            wn_id, cls_name = val[0], val[1]
            wnid_to_classes[wn_id] = cls_name

        return wnid_to_classes

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        data_dict = {
            "image": image,
            "target": target,
        }

        if self.return_contrastive_pairs:
            rank = self.ranks[index]
            max_rank = int(self.max_rank[self.targets[index]])
            pos_rank = int(min(rank + self.gap, max_rank))
            data_dict['pos_labels'] = self.targets[index]
            neg_idx = np.random.choice(self.indices_label_rank[self.targets[index]][rank])
            try:
                pos_idx = np.random.choice(self.indices_label_rank[self.targets[index]][pos_rank])
            except ValueError:
                found = False
                for fallback_rank in range(pos_rank + 1, max_rank + 1):
                    indices = self.indices_label_rank[self.targets[index]].get(fallback_rank, [])
                    if len(indices) > 0:
                        pos_idx = np.random.choice(indices)
                        found = True
                        break
                if not found:
                    pos_idx = -1
            if pos_idx != -1:
                image, _ = super().__getitem__(pos_idx)
                data_dict['positive'] = image
            else:
                return None
            image, _ = super().__getitem__(neg_idx)
            data_dict['negative'] = image
        return data_dict

    def set_num_group_and_group_array(self, num_shortcut_cat, shortcut_label):
        self.num_group = len(self.classes) * num_shortcut_cat
        self.group_array = (
            torch.tensor(self.targets, dtype=torch.long) * num_shortcut_cat
            + shortcut_label
        )

    def get_sampling_weights(self):
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array]
        return weights

class ImageNetPrecomputed():
    base_folder = "imagenet"

    def __init__(
        self,
        root: str,
        split: str = "train",
    ) -> None:
        assert split in ["train", "val"]
        with open(root, 'rb') as file:
            data = pickle.load(file)
        self.features = data['ftrs']
        self.labels = data['labels']

    def __len__(self):
        return len(self.features)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def load_meta_file(self, root):
        fpath = os.path.join(root, "labels.json")
        with open(fpath, 'r') as file:
            # Load the JSON data into a Python dictionary
            data = json.load(file)

        wnid_to_classes = {}

        for _, val in data.items():
            wn_id, cls_name = val[0], val[1]
            wnid_to_classes[wn_id] = cls_name

        return wnid_to_classes

    def __getitem__(self, index: int):
        features = self.features[index]
        target = self.labels[index]
        data_dict = {
            "image": features,
            "target": target,
            "index": index
        }
        return data_dict


def get_imagenet_class_name_list():
    with open("data/imagenet/labels.txt") as f:
        lines = f.readlines()

    prefix_len = len("n02892201,")
    class_name_list = [line[prefix_len:].strip() for line in lines]
    return class_name_list
