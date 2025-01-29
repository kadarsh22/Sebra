"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import os
import shutil
import numpy as np
import wandb
import pickle
import torch.nn as nn
from model.classifiers import get_classifier

from tqdm import tqdm
from .base_trainer import BaseTrainer
import torch.nn.functional as F
import torchvision
from utils import AverageMeter
from scipy.stats import kendalltau
from loss.upweighted_training_loss import UpweightedTrainingLoss
from loss.contrastive_loss import SupervisedContrastiveLoss


class OursTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        self.rank_order = {}
        args = self.args
        args.method = "ours"
        default_name = (
            f"{args.method}_{args.feature_extractor}"
            f"_lr_{args.lr:.0E}_{args.dataset}_lp"
        )
        self.default_name = default_name

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _before_train(self):
        self.criterion = UpweightedTrainingLoss(beta_inverse=.8)
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("cond_test/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("spuriosity_ranking/*", step_metric="rank")
        artifact = wandb.Artifact(wandb.run.name, type='model')

        self.train_set.return_contrastive_pairs = False
        e = 0

        while (self.train_loader.sampler.weights == 1).sum().item() != 0:
            try:
                self.rank_order[str(e)] = {label: [] for label in range(self.args.num_classes)}
                self.rank_order[str(e) + 'pr'] = {label: [] for label in range(self.args.num_classes)}
                log_dict = self.rank_spuriosity(e)
                e = e + 1
                wandb.log(log_dict)
            except KeyError:
                break


        # self.display_images_for_classes(self.rank_order, self.train_loader.dataset, e)
        ranks = [-1] * len(self.train_set)
        for rank, class_dict in self.rank_order.items():
            for class_index, indices in class_dict.items():
                for index in indices:
                    ranks[index] = int(rank)

        old_value = -1
        new_value = self.args.epoch + 1

        ranks = [new_value if x == old_value else x for x in ranks]

        print('Stage 1: Completed Spuriosity Ranking')

        rank_array = np.array(ranks)
        class_labels = np.array(self.train_set.targets)

        for class_id in range(1000):
            idx = np.where(class_labels == class_id)[0]
            per_class_rank = np.array(ranks)[idx]
            sorted_indices = np.argsort(per_class_rank)
            rank_order_list = np.array_split(idx[sorted_indices], 10)
            for rank, idx in enumerate(rank_order_list):
                rank_array[idx] = rank

        ranks = rank_array.tolist()

        self._setup_criterion()
        print('Creating non spurious training data')
        ranks_tensor = torch.tensor(ranks, device='cuda')
        class_labels = torch.tensor(self.train_set.targets, device='cuda')
        unique_labels = torch.unique(class_labels)
        unique_ranks = torch.unique(ranks_tensor)

        indices_by_label_rank = {}
        for label in unique_labels:
            label_mask = class_labels == label
            indices_by_label_rank[label.item()] = {}

            for rank in unique_ranks:
                rank_mask = ranks_tensor == rank
                condition = label_mask & rank_mask
                indices = torch.nonzero(condition, as_tuple=False).flatten()
                indices_by_label_rank[label.item()][rank.item()] = indices.cpu()

        self.train_loader_stage2.dataset.gap = self.args.gap
        self.train_loader_stage2.dataset.ranks = ranks
        self.train_loader_stage2.dataset.indices_label_rank = indices_by_label_rank
        self.train_loader_stage2.dataset.return_contrastive_pairs = True
        self.train_loader_stage2.dataset.max_rank = {outer_key: max(inner_dict) for outer_key, inner_dict in
                                                     indices_by_label_rank.items()}

    def rank_spuriosity(self, epoch):
        log_dict = self.train_erm(epoch)
        self.update_rank(epoch)
        return log_dict

    def train_erm(self, epoch):
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader,
                    dynamic_ncols=True, total=len(self.train_loader))
        for idx, (main_data) in enumerate(pbar):
            # ============= start: train classifier net ================
            self.classifier.train()
            img, label, idx_ = main_data['image'], main_data['target'], main_data['index']
            img = img.to(self.device)
            label = label.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.args.amp):
                target_logits = self.classifier(img.float())

                label = label.long()
                label = label.reshape(target_logits.shape[0])

                ce_loss = self.criterion(target_logits, label)
                ce_loss = ce_loss.mean()
                loss = ce_loss

            self.optimizer.zero_grad(set_to_none=True)
            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)

            losses.update(loss.item(), img.size(0))
            # ============= end: train classifier net ================

            self._scaler_update()  # activated only when using amp

            pbar.set_description('[{}/{}] ce: {:.3f}'.format(epoch, self.args.epoch, losses.avg, ))

        log_dict = {'ce_loss': losses.avg, }

        return log_dict

    def update_rank(self, epoch):
        old_weights_all = []
        new_weights_all = []
        targets_all = []
        index_all = []
        prob_all = []
        self.train_set.return_contrastive_pairs = False
        for data_dict in self.train_loader:
            image, target, index = data_dict["image"], data_dict["target"], data_dict['index']
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            index = index.to(self.device, non_blocking=True)
            with torch.no_grad():
                logit = self.classifier(image.float())

            p = F.softmax(logit.squeeze(1), dim=1)
            Yg = torch.gather(p, 1, torch.unsqueeze(target, 1))

            old_weights = self.train_loader.sampler.weights[index.cpu()]  # Assuming weights are on CPU
            threshold_mask = (Yg > self.args.threshold).long().squeeze(1)
            new_weights = old_weights - threshold_mask.cpu()
            old_weights_all.append(old_weights)
            new_weights_all.append(new_weights)
            targets_all.append(target)
            index_all.append(index)
            prob_all.append(Yg)
        old_weights = torch.stack(old_weights_all).view(-1)
        new_weights = torch.stack(new_weights_all).view(-1)
        target = torch.stack(targets_all).view(-1)
        index = torch.stack(index_all).view(-1)
        Yg = torch.stack(prob_all).view(-1)
        self.train_loader.sampler.weights[index.cpu()] = torch.clamp(new_weights, 0, 1)

        old_weights_gpu = old_weights.to(self.device, non_blocking=True)
        new_weights_gpu = new_weights.to(self.device, non_blocking=True)

        masks = [(target == label_temp) & (old_weights_gpu == 1) & (new_weights_gpu == 0) for label_temp in
                 range(self.args.num_classes)]

        indices = [index[mask].cpu().tolist() for mask in masks]  # Indices by class
        probabilities = [Yg[mask].cpu() for mask in masks]  # Probabilities by class

        for label_temp in range(self.args.num_classes):
            epoch_str = str(epoch)
            current_idx = indices[label_temp]
            current_pr = probabilities[label_temp]
            zipped_lists = zip(current_pr, current_idx)
            sorted_zipped = sorted(zipped_lists, reverse=True)
            current_idx_sorted = [element for _, element in sorted_zipped]
            self.rank_order[epoch_str][label_temp] = list(dict.fromkeys(current_idx_sorted))

        number_samples_in_train = (self.train_loader.sampler.weights == 1).sum().item()
        print(number_samples_in_train)

    def train(self):
        args = self.args
        self.classifier.train()
        losses = AverageMeter("Loss", ":.4e")
        classifier_losses = AverageMeter("classifier_Loss", ":.4e")
        contrastive_losses = AverageMeter("contrastive_loss", ":.4e")
        self.criterion = nn.CrossEntropyLoss()
        self.contrastive_loss = SupervisedContrastiveLoss(self.args.temperature)
        self.classifier.fc = nn.Identity()

        pbar = tqdm(self.train_loader_stage2, dynamic_ncols=True)
        for data_dict in pbar:
            image, image_pos, image_neg, image_pos_target = data_dict["image"], data_dict[
                'positive'], data_dict['negative'], data_dict['label']
            obj_gt = image_pos_target  # 0: obj, 1: bg, 2: co_occur_obj
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)
            image_pos = image_pos.to(self.device, non_blocking=True)
            with torch.no_grad():
                self.classifier.eval()
                features_pos = self.classifier(image_pos)
                features_neg = self.classifier(image_neg.to(self.device))
                self.classifier.train()

            with torch.cuda.amp.autocast(enabled=args.amp):
                image_ = torch.cat((image, image_pos))
                feature_ref = self.classifier(image_)
                output = self.classifier_head(feature_ref)
                contrastive_loss = self.contrastive_loss(feature_ref[:int(image_.shape[0] / 2)], features_pos.detach(),
                                                         features_neg.detach())
                classifier_loss = self.criterion(output[int(image_.shape[0] / 2):], obj_gt)
                loss = contrastive_loss + self.args.classifier_weight * classifier_loss

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), int(image.size(0) / 2))
            classifier_losses.update(classifier_loss.item(), int(image.size(0) / 2))
            contrastive_losses.update(contrastive_loss.item(), int(image.size(0) / 2))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f} classifier_loss: {classifier_losses.avg:.4f} "
                f"contrastive_loss: {contrastive_losses.avg:.4f}")
        log_dict = {'loss': losses.avg, 'classifier_loss': classifier_losses.avg, 'contrastive_loss': contrastive_losses.avg}
        return log_dict


    def display_images_for_classes(self, rank_order, dataset, cur_epoch, num_images_per_class=10, nrow=10):
        """
        Display images from a dataset for multiple classes based on rank order indices.

        Args:
            rank_order (dict): A dictionary where keys are strings representing classes, and values are lists of indices.
            dataset (Dataset): The dataset object from which images are to be fetched.
            num_images_per_class (int): Number of images to display from each class. Default is 10.
            nrow (int): Number of images per row in the grid. Default is 10.
        """
        # Loop through each class key and get the images
        for class_id in range(1, 6):
            i, j = 1, self.args.epoch
            top_ranked = []
            bottom_ranked = []
            while len(top_ranked) < 10:
                idx = list(set(rank_order[str(i)][class_id]))
                top_ranked.extend(idx)
                top_ranked = list(set(top_ranked))
                i = i + 1
            while len(bottom_ranked) < 10:
                idx = list(set(rank_order[str(j)][class_id]))
                bottom_ranked.extend(idx)
                bottom_ranked = list(set(bottom_ranked))
                j = j - 1
            images_high = torch.stack([dataset.__getitem__(x)['image'] for x in top_ranked][:num_images_per_class])
            images_low = torch.stack([dataset.__getitem__(x)['image'] for x in bottom_ranked][:num_images_per_class])

            combined_tensor = torch.cat((images_high, images_low), dim=0)
            grid = torchvision.utils.make_grid(combined_tensor, nrow=nrow, normalize=True)
            wandb.log({str(class_id): wandb.Image(grid)})

    def display_images_for_classes_spuriosity_ranking(self, dataset, num_images_per_class=10, nrow=10):
        """
        Display images from a dataset for multiple classes based on rank order indices.

        Args:
            rank_order (dict): A dictionary where keys are strings representing classes, and values are lists of indices.
            dataset (Dataset): The dataset object from which images are to be fetched.
            num_images_per_class (int): Number of images to display from each class. Default is 10.
            nrow (int): Number of images per row in the grid. Default is 10.
        """
        # Loop through each class key and get the images
        with open('assets/ranking/imagenet/spuriosity_ranking_imagenet.pkl', 'rb') as file:
            spuriosity_ranking_gt = pickle.load(file)
        for class_id in range(1, 6):
            try:
                rank_order = spuriosity_ranking_gt[class_id]['spurious']
                top_ranked = rank_order[:10].tolist()
                bottom_ranked = rank_order[-10:].tolist()
            except KeyError:
                rank_order = spuriosity_ranking_gt[class_id]['core']
                top_ranked = rank_order[:10].tolist()
                bottom_ranked = rank_order[-10:].tolist()

            images_high = torch.stack([dataset.__getitem__(x)['image'] for x in top_ranked])
            images_low = torch.stack([dataset.__getitem__(x)['image'] for x in bottom_ranked][:num_images_per_class])

            combined_tensor = torch.cat((images_high, images_low), dim=0)
            grid = torchvision.utils.make_grid(combined_tensor, nrow=nrow, normalize=True)
            wandb.log({str(class_id): wandb.Image(grid)})
