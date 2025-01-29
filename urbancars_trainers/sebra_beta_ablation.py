import torch
import re
from scipy.stats import spearmanr, weightedtau
from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
import glob
from model.classifiers import get_transforms
from dataset.urbancars import UrbanCars
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

rank_transform = get_transforms(arch='resnet50', is_training=False)
train_set = UrbanCars("/vol/research/project_storage/data/urbancars/", "train",
                      group_label='both', transform=rank_transform)

paths = glob.glob('exp/ablation/uc/ablation_varying_q/*')

values = []
for path in paths:
    # match = re.search(r'(\d+\.\d+)ablation\.pkl', path) ##threshold
    match = re.search(r'/([\d\.]+)_\d+\.\d+ablation\.pkl', path)  # parameter_q
    if match:
        ablating_parameter = match.group(1)
    else:
        raise NameError

    with open(path, 'rb') as file:
        spuriosity_ranking_ours = pickle.load(file)

    ranking_compiled = {i: [] for i in range(2)}
    for rank, rank_dict in spuriosity_ranking_ours.items():
        for i in range(2):
            class_wise_rank_order = ranking_compiled[i]
            class_wise_rank_order.extend(list(OrderedDict.fromkeys(rank_dict[i])))
            ranking_compiled[i] = class_wise_rank_order

    spuriosity_ranking_, ground_truth_ranking_, sizes = [], [], []

    ground_truth_ranking = torch.ones_like(train_set.obj_label == 0) * -1

    idx = (train_set.obj_label == 0) & ((train_set.bg_label == 0) & (train_set.co_occur_obj_label == 0)).bool()
    sizes.append(idx.sum().item())
    ground_truth_ranking[idx] = 0
    idx = (train_set.obj_label == 0) & ((train_set.bg_label == 0) & (train_set.co_occur_obj_label == 1)).bool()
    ground_truth_ranking[idx] = 1
    sizes.append(idx.sum().item())
    idx = (train_set.obj_label == 0) & ((train_set.bg_label == 1) & (train_set.co_occur_obj_label == 0)).bool()
    ground_truth_ranking[idx] = 2
    sizes.append(idx.sum().item())
    idx = (train_set.obj_label == 0) & ((train_set.bg_label == 1) & (train_set.co_occur_obj_label == 1)).bool()
    ground_truth_ranking[idx] = 3
    sizes.append(idx.sum().item())
    ground_truth_ranking_.append(ground_truth_ranking[:(train_set.obj_label == 0).sum().item()])
    split_indices = np.cumsum(sizes)[:-1]
    chunks = np.split(ranking_compiled[0], split_indices)
    spuriosity_ranking_ranks = [-1] * sum(sizes)
    for rank, chunk in enumerate(chunks):
        for idx in chunk:
            spuriosity_ranking_ranks[idx] = rank
    spuriosity_ranking_.append(spuriosity_ranking_ranks)

    ground_truth_ranking = torch.ones_like(train_set.obj_label == 1) * -1

    sizes = []
    idx = (train_set.obj_label == 1) & ((train_set.bg_label == 1) & (train_set.co_occur_obj_label == 1)).bool()
    ground_truth_ranking[idx] = 0
    sizes.append(idx.sum().item())

    idx = (train_set.obj_label == 1) & ((train_set.bg_label == 1) & (train_set.co_occur_obj_label == 0)).bool()
    ground_truth_ranking[idx] = 1
    sizes.append(idx.sum().item())

    idx = (train_set.obj_label == 1) & ((train_set.bg_label == 0) & (train_set.co_occur_obj_label == 1)).bool()
    ground_truth_ranking[idx] = 2
    sizes.append(idx.sum().item())

    idx = (train_set.obj_label == 1) & ((train_set.bg_label == 0) & (train_set.co_occur_obj_label == 0)).bool()
    ground_truth_ranking[idx] = 3
    sizes.append(idx.sum().item())
    ground_truth_ranking_.append(ground_truth_ranking[(train_set.obj_label == 0).sum().item():])
    split_indices = np.cumsum(sizes)[:-1]
    chunks = np.split(ranking_compiled[1], split_indices)
    min_index = min(ranking_compiled[1])
    spuriosity_ranking_ranks = [-1] * sum(sizes)
    for rank, chunk in enumerate(chunks):
        for idx in chunk:
            spuriosity_ranking_ranks[idx - min_index] = rank
    spuriosity_ranking_.append(spuriosity_ranking_ranks)

    rho_list = []
    rbo_avg = []
    kendal_all = []
    for ranking1, ranking2 in zip(spuriosity_ranking_, ground_truth_ranking_):
        rho, p_value = spearmanr(ranking1, ranking2)
        rho_list.append(rho)
        kendal_all.append(weightedtau(ranking1, ranking2).statistic)
    kendals_tau = sum(kendal_all) / len(kendal_all)
    values.append((float(ablating_parameter), kendals_tau))

sorted_data = sorted(values, key=lambda x: x[0])
x_values, y_values = zip(*sorted_data)
beta_values = tuple(1 / x for x in x_values)

plt.figure(figsize=(6, 4))

# Line plot with markers for better readability, colorblind-friendly colors
plt.plot(beta_values, y_values, color='#0072B2', linestyle='-', linewidth=2, marker='o', markersize=8,
         label='Trend Line', markerfacecolor='white', markeredgewidth=2)

# Add labels with Greek letters for beta and tau
plt.xlabel(r'$\beta$', fontsize=16)  # Greek symbol for beta
plt.ylabel(r"Kendall's $\tau$ Coefficient", fontsize=16)  # Greek symbol for tau

# Set axis limits for better focus
plt.xlim(0.5, 5.5)  # Slightly beyond the data range
plt.ylim(0.78, 0.87)

# Set tick sizes for readability
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(which='both', width=1)  # Adjust tick width for major and minor ticks
plt.tick_params(which='minor', length=4, color='gray')

# Add grid with subtle lines and minor grid
plt.grid(True, linestyle='--', alpha=0.6)
plt.grid(True, which='minor', linestyle=':', alpha=0.3)
plt.tight_layout(pad=0)
plt.savefig('assets/plot_ablation_beta_inverse.pdf', format='pdf', bbox_inches='tight', pad_inches=0, dpi=1200, transparent=True)

# Display the plot
plt.show()
