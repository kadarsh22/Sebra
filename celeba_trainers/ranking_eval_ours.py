import torch
import numpy as np
import pickle
import json
from scipy.stats import spearmanr, weightedtau
from collections import OrderedDict
from dataset.multiceleba import BiasedCelebA
from model.classifiers import get_transforms

rank_transform = get_transforms(arch='resnet50', is_training=False)
with open('exp/ranks_celeba/0.7_0.7_0.001_0.001_0.9_256_rank_order_celeba_ours.pkl', 'rb') as file:
    spuriosity_ranking = pickle.load(file)

ranking_compiled = {i: [] for i in range(2)}
for rank, rank_dict in spuriosity_ranking.items():
    for i in range(2):
        class_wise_rank_order = ranking_compiled[i]
        class_wise_rank_order.extend(list(OrderedDict.fromkeys(rank_dict[i])))
        ranking_compiled[i] = class_wise_rank_order

transform = get_transforms('resnet', is_training=True, dataset='celebA')
idx2attr = json.load(open("create_datasets/celeba/idx2attr.json", 'r'))
idx2attr = {int(k): v for k, v in idx2attr.items()}
target_name = idx2attr[31]
biasA_name = idx2attr[20]
biasB_name = idx2attr[39]
train_set = BiasedCelebA(root='data/', target_name=target_name, biasA_name=biasA_name, biasB_name=biasB_name,
                         biasA_ratio=0.95, biasB_ratio=0.95, split='train',
                         transform=transform)

spuriosity_ranking_ = []
ground_truth_ranking_ = []
sizes = []

ground_truth_ranking = train_set.obj_label == 0
zero_class = np.where(ground_truth_ranking)[0]
init_ranks = [-1] * len(zero_class)
ground_truth_ranking = dict(zip(zero_class, init_ranks))

sizes = []
bias_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]

for rank, (bias_a, bias_b) in enumerate(bias_combinations):
    idx_ = torch.tensor((train_set.obj_label == 0) &
                        (train_set.bias_a_label == bias_a) &
                        (train_set.bias_b_label == bias_b)).bool()
    idx = np.where(idx_)[0]
    for idx_temp in idx:
        ground_truth_ranking[idx_temp] = rank
    sizes.append(idx_.sum().item())

ground_truth_ranking_.append(ground_truth_ranking)


split_indices = np.cumsum(sizes)[:-1]
chunks = np.split(ranking_compiled[0], split_indices)
spuriosity_ranking_ranks = dict(zip(zero_class, init_ranks))
for rank, chunk in enumerate(chunks):
    for idx in chunk:
        spuriosity_ranking_ranks[idx] = rank
spuriosity_ranking_.append(spuriosity_ranking_ranks)

ground_truth_ranking = train_set.obj_label == 1
one_class = np.where(ground_truth_ranking == True)[0]
init_ranks = [-1] * len(one_class)
ground_truth_ranking = dict(zip(one_class, init_ranks))

sizes = []
bias_combinations = [(1, 1), (1, 0), (0, 1), (0, 0)]

for rank, (bias_a, bias_b) in enumerate(bias_combinations):
    idx_ = torch.tensor((train_set.obj_label == 1) &
                        (train_set.bias_a_label == bias_a) &
                        (train_set.bias_b_label == bias_b)).bool()
    idx = np.where(idx_)[0]
    for idx_temp in idx:
        ground_truth_ranking[idx_temp] = rank
    sizes.append(idx_.sum().item())
ground_truth_ranking_.append(ground_truth_ranking)

split_indices = np.cumsum(sizes)[:-1]
chunks = np.split(ranking_compiled[1], split_indices)
spuriosity_ranking_ranks = dict(zip(one_class, init_ranks))
for rank, chunk in enumerate(chunks):
    for idx in chunk:
        spuriosity_ranking_ranks[idx] = rank
spuriosity_ranking_.append(spuriosity_ranking_ranks)

rbo_avg = []
rho_list = []
tau_avg  = []
for ranking1, ranking2 in zip(spuriosity_ranking_, ground_truth_ranking_):
    common_keys = set(ranking1.keys()) & set(ranking2.keys())
    ranks_a = [ranking1[key] for key in common_keys]
    ranks_b = [ranking2[key] for key in common_keys]
    rho, p_value = spearmanr(ranks_a, ranks_b)
    tau_avg.append(weightedtau(ranks_a, ranks_b).statistic)
    rho_list.append(rho)

print(sum(rho_list) / len(rho_list))
print(sum(tau_avg) / len(tau_avg))
