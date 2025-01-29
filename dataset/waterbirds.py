import os
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
#


class WaterbirdDataset(Dataset):
    def __init__(self, data_correlation, split, root_dir='/mnt/fast/nobackup/users/ak03476/data/waterbirds'):
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        self.env_dict = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (1, 1): 3
        }
        self.split = split
        self.root_dir = root_dir
        self.dataset_name = "waterbird_complete" + "{:0.2f}".format(data_correlation)[-2:] + "_forest2water2"
        self.dataset_dir = os.path.join(self.root_dir, self.dataset_name)
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, 'metadata.csv'))
        self.metadata_df = self.metadata_df[self.metadata_df['split'] == self.split_dict[self.split]]

        self.y_array = self.metadata_df['y'].values
        self.place_array = self.metadata_df['place'].values
        self.filename_array = self.metadata_df['img_filename'].values
        self.transform = get_transform_cub(self.split == split)
        self.return_contrastive_pairs = False
        self.imgs = [Image.open(os.path.join(self.dataset_dir, filename)).convert('RGB') for filename in self.filename_array.tolist()]

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        place = self.place_array[idx]
        img = self.transform(self.imgs[idx])
        if self.return_contrastive_pairs:
            data_dict = {"image": img, "label": y, "index": idx}
            rank = self.ranks[idx]
            max_rank = 46
            if y == 0:
                gap = 33
            else:
                gap = 22
            pos_rank = int(min(rank + gap, max_rank))
            data_dict['pos_labels'] = self.y_array[idx]
            neg_idx = np.random.choice(self.indices_label_rank[self.y_array[idx].item()][rank])
            try:
                pos_idx = np.random.choice(self.indices_label_rank[self.y_array[idx].item()][pos_rank])
            except:
                found = False
                for fallback_rank in range(pos_rank + 1, max_rank + 1):
                    indices = self.indices_label_rank[self.y_array[idx].item()].get(fallback_rank, [])
                    if len(indices) > 0:
                        pos_idx = np.random.choice(indices)
                        found = True
                        break
                if not found:
                    pos_idx = -1
            if pos_idx != -1:
                data_dict['positive'] = self.transform(self.imgs[pos_idx])
            else:
                return None
            x_neg = self.imgs[neg_idx]
            data_dict['negative'] = self.transform(x_neg)
            data_dict['idx'] = idx
            data_dict['pos_idx'] = pos_idx
            data_dict['neg_idx'] = neg_idx
            return data_dict

        return img, (y, place), self.env_dict[(y, place)] , idx


def get_transform_cub(train):
    scale = 256.0 / 224.0
    target_resolution = (224, 224)
    assert target_resolution is not None

    if not train:
        # Resizes the image to a slightly larger square then crops the center.
        # transform = transforms.Compose([
        #     transforms.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
        #     transforms.CenterCrop(target_resolution),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(
        #         target_resolution,
        #         scale=(0.7, 1.0),
        #         ratio=(0.75, 1.3333333333333333),
        #         interpolation=2),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform


def get_waterbird_dataloader(batch_size, data_label_correlation, split):
    kwargs = {'pin_memory': False, 'num_workers': 0, 'drop_last': True}
    dataset = WaterbirdDataset(data_correlation=data_label_correlation, split=split)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            **kwargs)
    return dataloader