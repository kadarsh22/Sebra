import torch
import numpy as np
import pickle
from scipy.stats import weightedtau
from model.classifiers import get_transforms
from dataset.urbancars import UrbanCars

rank_transform = get_transforms(arch='resnet50', is_training=False)
with open('assets/ranking/urbancars/neurips23mmoayeri/spu_rankings_UC_nips.pkl', 'rb') as file:
    spuriosity_ranking_org = pickle.load(file)

spuriosity_ranking = spuriosity_ranking_org['top_idx']['train']
train_set = UrbanCars("/vol/research/project_storage/data/urbancars/", "train",
                      group_label='both', transform=rank_transform)

spuriosity_ranking_ = []
ground_truth_ranking_ = []
sizes = []

ground_truth_ranking = torch.ones_like(train_set.obj_label == 0) * -1

# Define the combinations of bg_label and co_occur_obj_label
combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
sizes = []
ground_truth_ranking = torch.ones_like(train_set.obj_label == 0) * -1

# Loop through combinations and set ground_truth_ranking and sizes
for rank, (bg, co_obj) in enumerate(combinations):
    idx = (train_set.obj_label == 0) & (train_set.bg_label == bg) & (train_set.co_occur_obj_label == co_obj).bool()
    sizes.append(idx.sum().item())
    ground_truth_ranking[idx] = rank

ground_truth_ranking_.append(ground_truth_ranking[:(train_set.obj_label == 0).sum().item()])
split_indices = np.cumsum(sizes)[:-1]
chunks = np.split(np.flip(spuriosity_ranking[0]), split_indices)

# Assign ranks to spuriosity_ranking_ranks
spuriosity_ranking_ranks = [-1] * sum(sizes)
for rank, chunk in enumerate(chunks):
    for idx in chunk:
        spuriosity_ranking_ranks[idx] = rank

spuriosity_ranking_.append(spuriosity_ranking_ranks)

combinations = [(1, 1), (1, 0), (0, 1), (0, 0)]
sizes = []
ground_truth_ranking = torch.ones_like(train_set.obj_label == 1) * -1

# Loop through combinations and set ground_truth_ranking and sizes
for rank, (bg, co_obj) in enumerate(combinations):
    idx = (train_set.obj_label == 1) & (train_set.bg_label == bg) & (train_set.co_occur_obj_label == co_obj).bool()
    sizes.append(idx.sum().item())
    ground_truth_ranking[idx] = rank

ground_truth_ranking_.append(ground_truth_ranking[(train_set.obj_label == 0).sum().item():])

# Split spuriosity ranking based on sizes
split_indices = np.cumsum(sizes)[:-1]
chunks = np.split(np.flip(spuriosity_ranking[1]), split_indices)

# Assign ranks to spuriosity_ranking_ranks
min_index = spuriosity_ranking[1].min()
spuriosity_ranking_ranks = [-1] * sum(sizes)
for rank, chunk in enumerate(chunks):
    for idx in chunk:
        spuriosity_ranking_ranks[idx - min_index] = rank

spuriosity_ranking_.append(spuriosity_ranking_ranks)

kendal_all = []
for ranking1, ranking2 in zip(spuriosity_ranking_, ground_truth_ranking_):
    kendal_all.append(weightedtau(ranking1, ranking2).statistic)
print(sum(kendal_all) / len(kendal_all))
