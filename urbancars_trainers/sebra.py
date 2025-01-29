import torch
import wandb
import torchvision
import numpy as np
from tqdm import tqdm
from .base_trainer import BaseTrainer
import torch.nn.functional as F
from utils import AverageMeter
import torch.nn as nn
from loss.upweighted_training_loss import UpweightedTrainingLoss
from loss.contrastive_loss import SupervisedContrastiveLoss


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    image = torch.stack([item['image'] for item in batch])
    positives = torch.stack([item['positive'] for item in batch])
    negatives = torch.stack([item['negative'] for item in batch])
    labels = torch.tensor([item['pos_labels'] for item in batch])

    return {
        'image': image,
        'positive': positives,
        'negative': negatives,
        'label': labels
    }


class SebraTrainer(BaseTrainer):
    def _setup_method_name_and_default_name(self):
        default_name = '{}_{}_lr_{:.0E}_beta_{:.2E}_lambda_{:.0E}'.format(
            self.args.method,
            self.args.dataset_name,
            self.args.lr,
            self.args.beta_inverse,
            self.args.p_critical
            )
        self.default_name = default_name
        self.rank_order = {}

    def _before_train(self):
        self.criterion = UpweightedTrainingLoss(beta_inverse=self.args.beta_inverse)
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("cond_test/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("spuriosity_ranking/*", step_metric="rank")

        self.train_set.return_contrastive_pairs = False

        ranks = [-1] * len(self.train_set)
        for e in range(self.cur_epoch, self.args.epoch + 1):
            try:
                self.rank_order[str(e)] = {label: [] for label in range(self.args.num_classes)}
                log_dict = self.rank_spuriosity()
                log_dict.update({"rank": e})
                wandb.log(log_dict)
            except RuntimeError as error:
                print(error)
                self.display_images_for_classes(self.rank_order, self.train_set, e)
                for rank, class_dict in self.rank_order.items():
                    for class_index, indices in class_dict.items():
                        for index in indices:
                            ranks[index] = int(rank)
                break
            self.cur_epoch += 1
   
        for rank, class_dict in self.rank_order.items():
            for class_index, indices in class_dict.items():
                for index in indices:
                    ranks[index] = int(rank)

        print('Stage 1: Completed Spuriosity Ranking')

        self._setup_dataset()
        self._setup_criterion()
        self._setup_models_stage2()
        self._setup_optimizers_stage2()

        print('Creating non spurious training data')
        class_labels = self.train_set.obj_bg_co_occur_obj_label_list[:, 0]
        unique_labels = np.unique(class_labels)

        indices_by_label_rank = {}
        for label in unique_labels:  # Precompute indices by label and rank for efficiency
            indices_by_label_rank[label] = {}
            for rank in np.unique(ranks):
                idx_ = np.where((class_labels == label) & (ranks == rank))[0]
                np.random.shuffle(idx_)
                indices_by_label_rank[label][rank] = idx_

        self.train_set.ranks = ranks
        self.train_set.indices_label_rank = indices_by_label_rank
        self.train_set.return_contrastive_pairs = True
        self.train_set.max_rank = {outer_key: max(inner_dict) for outer_key, inner_dict in
                                   indices_by_label_rank.items()}
        self.train_set.gap = self.args.gap

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.args.batch_size_stage2,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            persistent_workers=self.args.num_workers > 0,
            collate_fn=collate_fn
        )
        self.cond_best_acc = 0
        self.cond_on_best_val_log_dict = {}
        self.cur_epoch = 1

    def _set_train(self):
        self.classifier.eval()

    def rank_spuriosity(self):
        log_dict = self.train_erm()
        count_dict = self.update_rank(self.cur_epoch)
        log_dict.update(count_dict)
        return log_dict

    def train_erm(self):
        args = self.args
        self.classifier.train()
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = self.classifier(image)
                loss = self.criterion(output, obj_gt).mean()

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), image.size(0))

            pbar.set_description(
                f"ERM: [{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f}"
            )
        log_dict = {'ce_loss': losses.avg}
        return log_dict

    def train(self):
        args = self.args
        self._set_train()
        losses = AverageMeter("Loss", ":.4e")
        classifier_losses = AverageMeter("classifier_Loss", ":.4e")
        contrastive_losses = AverageMeter("contrastive_loss", ":.4e")
        self.classifier_head.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        contrastive_loss_criterion = SupervisedContrastiveLoss(self.args.temperature)
        self.classifier.fc = nn.Identity()

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, image_pos, image_neg, image_pos_target = data_dict["image"], data_dict[
                'positive'], data_dict['negative'], data_dict['label']
            obj_gt = image_pos_target  # 0: obj, 1: bg, 2: co_occur_obj
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)
            image_pos = image_pos.to(self.device, non_blocking=True)
            with torch.no_grad():
                self.classifier.eval()
                features_pos = self.classifier(image_pos.to(self.device))
                features_neg = self.classifier(image_neg.to(self.device))
                self.classifier.train()

            with torch.cuda.amp.autocast(enabled=args.amp):
                image_ = torch.cat((image, image_pos))
                feature_ref = self.classifier(image_)
                output = self.classifier_head(feature_ref)
                contrastive_loss = contrastive_loss_criterion(feature_ref[:int(image_.shape[0] / 2)], features_pos.detach(),
                                                         features_neg.detach())
                classifier_loss = criterion(output[int(image_.shape[0] / 2):], obj_gt)
                loss = contrastive_loss + self.args.classifier_weight * classifier_loss

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), int(image.size(0)/2))
            classifier_losses.update(classifier_loss.item(), int(image.size(0) / 2))
            contrastive_losses.update(contrastive_loss.item(), int(image.size(0) / 2))

            pbar.set_description(
                f"[{self.cur_epoch}/{args.epoch}] loss: {losses.avg:.4f} classifier_loss: {classifier_losses.avg:.4f} "
                f"contrastive_loss: {contrastive_losses.avg:.4f}")

    def update_rank(self, epoch):
        for data_dict in self.train_loader:
            image, target, index = data_dict["image"], data_dict["label"], data_dict['index']
            obj_gt = target[:, 0]  # 0: obj, 1: bg, 2: co_occur_obj
            image = image.to(self.device, non_blocking=True)
            target = obj_gt.to(self.device, non_blocking=True)
            with torch.no_grad():
                logit = self.classifier(image)
            p = F.softmax(logit.squeeze(1), dim=1)
            Yg = torch.gather(p, 1, torch.unsqueeze(target, 1))
            old_weights = self.train_loader.sampler.weights[index]
            new_weights = old_weights - (Yg > self.args.p_critical).long().squeeze(1).cpu()
            self.train_loader.sampler.weights[index] = torch.clamp(new_weights, 0, 1)
            index.cuda()
            for label_temp in range(self.args.num_classes):
                epoch_str = str(epoch)
                current_idx = self.rank_order[epoch_str][label_temp]
                mask = (target.cuda() == label_temp) & (old_weights.cuda() == 1) & (new_weights.cuda() == 0)
                current_idx.extend(index[mask.cpu()].tolist())
                self.rank_order[epoch_str][label_temp] = current_idx
        data_counts = self._visualise_weighted_points()
        return data_counts

    def _visualise_weighted_points(self):
        weight_ones = torch.where(self.train_loader.sampler.weights == 1)[0]
        data_ones = [self.train_set.__getitem__(idx) for idx in weight_ones]
        image_dict = {}
        for x, plot_name in zip([data_ones], ['weight_ones']):
            x_values = torch.stack([d['label'] for d in x])
            obj_attr_values = x_values[:, 0]
            bg_values = x_values[:, 1]
            coobj_values = x_values[:, 2]

            num_aligned_aligned = torch.sum((obj_attr_values == bg_values) & (obj_attr_values == coobj_values)).item()
            num_aligned_conflicting = torch.sum(
                (obj_attr_values == bg_values) & (obj_attr_values != coobj_values)).item()
            num_conflict_aligned = torch.sum((obj_attr_values != bg_values) & (obj_attr_values == coobj_values)).item()
            num_conflict_conflict = torch.sum((obj_attr_values != bg_values) & (obj_attr_values != coobj_values)).item()

            image_dict['spuriosity_ranking/' + plot_name + '_bg_align_coobj_align'] = num_aligned_aligned
            image_dict['spuriosity_ranking/' + plot_name + '_bg_align_coobj_conflict'] = num_aligned_conflicting
            image_dict['spuriosity_ranking/' + plot_name + '_bg_conflict_coobj_align'] = num_conflict_aligned
            image_dict['spuriosity_ranking/' + plot_name + '_bg_conflict_coobj_conflict'] = num_conflict_conflict
        return image_dict

    def display_images_for_classes(self, rank_order, dataset, cur_epoch, num_images_per_class=10, nrow=10):

        """
        Display images from a dataset for multiple classes based on rank order indices.

        Args:
            rank_order (dict): A dictionary where keys are strings representing classes, and values are lists of indices.
            dataset (Dataset): The dataset object from which images are to be fetched.
            num_images_per_class (int): Number of images to display from each class. Default is 10.
            nrow (int): Number of images per row in the grid. Default is 10.
        """
        for class_id in range(self.args.num_classes):
            i, j = 1, cur_epoch
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
            images_high = torch.stack([dataset.__getitem__(x)["image"] for x in top_ranked][:num_images_per_class])
            images_low = torch.stack([dataset.__getitem__(x)["image"] for x in bottom_ranked][:num_images_per_class])

            combined_tensor = torch.cat((images_high, images_low), dim=0)
            grid = torchvision.utils.make_grid(combined_tensor, nrow=nrow, normalize=True)
            wandb.log({'class' + str(class_id): wandb.Image(grid)})
