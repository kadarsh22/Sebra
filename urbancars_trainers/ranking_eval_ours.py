import torch
import rbo
from scipy.stats import spearmanr, weightedtau
from collections import OrderedDict
import pickle
import glob
from scipy.stats import kendalltau
import torchvision
from model.classifiers import get_transforms
from dataset.urbancars import UrbanCars
import numpy as np
import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

rank_transform = get_transforms(arch='resnet50', is_training=False)
with open('exp/ranks_uc/0.7_0.8_0.001_0.001_128_rank_order_uc_ours.pkl', 'rb') as file:
    spuriosity_ranking_ours = pickle.load(file)

ranking_compiled = {i: [] for i in range(2)}
for rank, rank_dict in spuriosity_ranking_ours.items():
    for i in range(2):
        class_wise_rank_order = ranking_compiled[i]
        class_wise_rank_order.extend(list(OrderedDict.fromkeys(rank_dict[i])))
        ranking_compiled[i] = class_wise_rank_order

train_set = UrbanCars("/vol/research/project_storage/data/urbancars/", "train",
                      group_label='both', transform=rank_transform)

spuriosity_ranking_ = []
ground_truth_ranking_ = []
sizes = []

# Define combinations of bg_label and co_occur_obj_label for obj_label == 0
combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]

ground_truth_ranking = torch.ones_like(train_set.obj_label == 0) * -1
sizes = []

# Loop through combinations to set ground_truth_ranking and sizes
for rank, (bg, co_obj) in enumerate(combinations):
    idx = (train_set.obj_label == 0) & (train_set.bg_label == bg) & (train_set.co_occur_obj_label == co_obj).bool()
    ground_truth_ranking[idx] = rank
    sizes.append(idx.sum().item())

# Append the ground truth ranking to the list
ground_truth_ranking_.append(ground_truth_ranking[:(train_set.obj_label == 0).sum().item()])

# Split spuriosity ranking based on sizes
split_indices = np.cumsum(sizes)[:-1]
chunks = np.split(ranking_compiled[0], split_indices)

# Assign ranks to spuriosity_ranking_ranks
spuriosity_ranking_ranks = [-1] * sum(sizes)
for rank, chunk in enumerate(chunks):
    for idx in chunk:
        spuriosity_ranking_ranks[idx] = rank

spuriosity_ranking_.append(spuriosity_ranking_ranks)

combinations = [(1, 1), (1, 0), (0, 1), (0, 0)]

ground_truth_ranking = torch.ones_like(train_set.obj_label == 1) * -1
sizes = []

# Loop through combinations to set ground_truth_ranking and sizes
for rank, (bg, co_obj) in enumerate(combinations):
    idx = (train_set.obj_label == 1) & (train_set.bg_label == bg) & (train_set.co_occur_obj_label == co_obj).bool()
    ground_truth_ranking[idx] = rank
    sizes.append(idx.sum().item())

# Append the ground truth ranking to the list for obj_label == 1
ground_truth_ranking_.append(ground_truth_ranking[(train_set.obj_label == 0).sum().item():])

# Split spuriosity ranking based on sizes
split_indices = np.cumsum(sizes)[:-1]
chunks = np.split(ranking_compiled[1], split_indices)

# Assign ranks to spuriosity_ranking_ranks
min_index = min(ranking_compiled[1])
spuriosity_ranking_ranks = [-1] * sum(sizes)
for rank, chunk in enumerate(chunks):
    for idx in chunk:
        spuriosity_ranking_ranks[idx - min_index] = rank

spuriosity_ranking_.append(spuriosity_ranking_ranks)

kendal_all = []
for ranking1, ranking2 in zip(spuriosity_ranking_, ground_truth_ranking_):
    kendal_all.append(weightedtau(ranking1, ranking2).statistic)
print(sum(kendal_all) / len(kendal_all))