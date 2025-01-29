import torch
import rbo
from scipy.stats import spearmanr, weightedtau
from collections import OrderedDict
import pickle
import json
from dataset.multiceleba import BiasedCelebA
from model.classifiers import get_transforms
import numpy as np

rank_transform = get_transforms(arch='resnet50', is_training=False)
with open('assets/ranking/celeba/neurips23mmoayeri/spu_rankings_celeba_nips_more_feat_1.pkl', 'rb') as file:
    spuriosity_ranking = pickle.load(file)['bot_idx']['train']

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
zero_class = np.where(ground_truth_ranking == True)[0]
init_ranks = [-1] * len(zero_class)
ground_truth_ranking = dict(zip(zero_class, init_ranks))

idx_ = torch.tensor((train_set.obj_label == 0) & ((train_set.bias_a_label == 0) & (train_set.bias_b_label == 0))).bool()
idx = np.where(idx_ == True)[0]
for idx_temp in idx:
    ground_truth_ranking[idx_temp] = 0
sizes.append(idx_.sum().item())

idx_ = torch.tensor((train_set.obj_label == 0) & ((train_set.bias_a_label == 0) & (train_set.bias_b_label == 1))).bool()
idx = np.where(idx_ == True)[0]
for idx_temp in idx:
    ground_truth_ranking[idx_temp] = 1
sizes.append(idx_.sum().item())

idx_ = torch.tensor((train_set.obj_label == 0) & ((train_set.bias_a_label == 1) & (train_set.bias_b_label == 0))).bool()
idx = np.where(idx_ == True)[0]
for idx_temp in idx:
    ground_truth_ranking[idx_temp] = 2
sizes.append(idx_.sum().item())

idx_ = torch.tensor((train_set.obj_label == 0) & ((train_set.bias_a_label == 1) & (train_set.bias_b_label == 1))).bool()
idx = np.where(idx_ == True)[0]
for idx_temp in idx:
    ground_truth_ranking[idx_temp] = 3
sizes.append(idx_.sum().item())

ground_truth_ranking_.append(ground_truth_ranking)

split_indices = np.cumsum(sizes)[:-1]
chunks = np.split(spuriosity_ranking[0], split_indices)
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
idx_ = torch.tensor((train_set.obj_label == 1) & ((train_set.bias_a_label == 1) & (train_set.bias_b_label == 1))).bool()
idx = np.where(idx_ == True)[0]
for idx_temp in idx:
    ground_truth_ranking[idx_temp] = 0
sizes.append(idx_.sum().item())

idx_ = torch.tensor((train_set.obj_label == 1) & ((train_set.bias_a_label == 1) & (train_set.bias_b_label == 0))).bool()
idx = np.where(idx_ == True)[0]
for idx_temp in idx:
    ground_truth_ranking[idx_temp] = 1
sizes.append(idx_.sum().item())

idx_ = torch.tensor((train_set.obj_label == 1) & ((train_set.bias_a_label == 0) & (train_set.bias_b_label == 1))).bool()
idx = np.where(idx_ == True)[0]
for idx_temp in idx:
    ground_truth_ranking[idx_temp] = 2
sizes.append(idx_.sum().item())

idx_ = torch.tensor((train_set.obj_label == 1) & ((train_set.bias_a_label == 0) & (train_set.bias_b_label == 0))).bool()
idx = np.where(idx_ == True)[0]
for idx_temp in idx:
    ground_truth_ranking[idx_temp] = 3
sizes.append(idx_.sum().item())

ground_truth_ranking_.append(ground_truth_ranking)

split_indices = np.cumsum(sizes)[:-1]
chunks = np.split(spuriosity_ranking[1], split_indices)
spuriosity_ranking_ranks = dict(zip(one_class, init_ranks))
for rank, chunk in enumerate(chunks):
    for idx in chunk:
        spuriosity_ranking_ranks[idx] = rank
spuriosity_ranking_.append(spuriosity_ranking_ranks)

rbo_avg = []
rho_list = []
kendalltau_avg = []
for ranking1, ranking2 in zip(spuriosity_ranking_, ground_truth_ranking_):
    common_keys = set(ranking1.keys()) & set(ranking2.keys())
    ranks_a = [ranking1[key] for key in common_keys]
    ranks_b = [ranking2[key] for key in common_keys]
    rho, p_value = spearmanr(ranks_a, ranks_b)
    rho_list.append(rho)
    kendalltau_avg.append(weightedtau(ranks_b, ranks_a).statistic)
print(kendalltau_avg)
print(sum(kendalltau_avg) / len(kendalltau_avg))
