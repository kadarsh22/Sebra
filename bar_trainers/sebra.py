import os
import torch
import wandb
import torchvision
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from utils import AverageMeter
from loss.upweighted_training_loss import UpweightedTrainingLoss
from loss.contrastive_loss import SupervisedContrastiveLoss
from bar_trainers.bar_args import get_parser, parse_and_check
from bar_trainers.base_trainer import BaseActionTrainer
from model.model_zoo import get_model_and_transforms
from common.utils import initialize_seeds, collate_fn


class Trainer(BaseActionTrainer):
    def __init__(self, trainer_args):
        super(Trainer, self).__init__(trainer_args)
        self.args = trainer_args
        self.rank_order = {}
        self._modify_rank_loader()

    def _setup_models(self):
        self.model, _ = get_model_and_transforms(args.arch, num_classes=self.args.num_classes)
        self.classifier_head = nn.Identity()

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def _setup_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.lr, weight_decay=args.weight_decay)

    def _setup_models_stage2(self):
        self.model, _ = get_model_and_transforms(self.args.arch, num_classes=self.args.num_classes, pretrained=True)
        self.classifier_head = nn.Linear(self.model.fc.in_features, self.args.num_classes).to(self.device)
        self.model.fc = torch.nn.Identity()
        for p in self.model.fc.parameters():
            p.requires_grad = False

    def _setup_optimizers_stage2(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad] + list(
            self.classifier_head.parameters())
        self.optimizer = torch.optim.Adam(parameters, self.args.lr)

    def _before_train(self):
        self.criterion = UpweightedTrainingLoss(beta_inverse=self.args.beta_inverse)
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("cond_test/*", step_metric="epoch")
        wandb.define_metric("test/*", step_metric="epoch")
        wandb.define_metric("spuriosity_ranking/*", step_metric="rank")

        self.train_dataset.return_contrastive_pairs = False

        ranks = [-1] * len(self.train_dataset)
        for e in range(1, self.args.epoch + 1):
            try:
                self.rank_order[str(e)] = {label: [] for label in range(self.args.num_classes)}
                log_dict = self.rank_spuriosity(e)
                wandb.log(log_dict)
            except RuntimeError as error:
                print(error)
                self.display_images_for_classes(self.rank_order, self.rank_loader.dataset, e)
                for rank, class_dict in self.rank_order.items():
                    for class_index, indices in class_dict.items():
                        for index in indices:
                            ranks[index] = int(rank)
                break

        print('Stage 1: Completed Spuriosity Ranking')

        self._setup_criterion()
        self._setup_models_stage2()
        self._setup_optimizers_stage2()

        class_labels = self.train_dataset.label_list
        unique_labels = np.unique(class_labels)

        indices_by_label_rank = {}
        for label in unique_labels:  # Precompute indices by label and rank for efficiency
            indices_by_label_rank[label] = {}
            for rank in np.unique(ranks):
                idx_ = np.where((class_labels == label) & (ranks == rank))[0]
                np.random.shuffle(idx_)
                indices_by_label_rank[label][rank] = idx_

        self.train_dataset.ranks = ranks
        self.train_dataset.indices_label_rank = indices_by_label_rank
        self.train_dataset.return_contrastive_pairs = True
        self.train_dataset.max_rank = {outer_key: max(inner_dict) for outer_key, inner_dict in
                                       indices_by_label_rank.items()}
        self.train_dataset.gap = self.args.gap

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size_stage2,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_fn)

    def _setup_method_name_and_default_name(self):
        self.default_name = (
            f"{self.args.method_name}_{self.args.dset_name}_"
            f"seed_{self.args.seed}_"
            f"lr_{self.args.lr:.0E}_"
            f"beta_{self.args.beta_inverse:.0E}_"
            f"pcritical_{self.args.p_critical:.2E}"
        )

        return self.default_name

    def rank_spuriosity(self, epoch):
        log_dict = self.train_erm(epoch)
        self.update_rank(epoch)
        return log_dict

    def train(self, epoch):
        losses = AverageMeter("Loss", ":.4e")
        classifier_losses = AverageMeter("classifier_Loss", ":.4e")
        contrastive_losses = AverageMeter("contrastive_loss", ":.4e")
        self.classifier_head.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        contrastive_loss_criterion = SupervisedContrastiveLoss(self.args.temperature)
        self.model.fc = nn.Identity()

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, image_pos, image_neg, image_pos_target = data_dict["image"], data_dict[
                'positive'], data_dict['negative'], data_dict['label']
            obj_gt = image_pos_target
            image = image.to(self.device, non_blocking=True)
            obj_gt = obj_gt.to(self.device, non_blocking=True)
            image_pos = image_pos.to(self.device, non_blocking=True)
            with torch.no_grad():
                self.model.eval()
                features_pos = self.model(image_pos.to(self.device))
                features_neg = self.model(image_neg.to(self.device))
                self.model.train()

            with torch.cuda.amp.autocast(enabled=self.args.amp):
                image_ = torch.cat((image, image_pos))
                feature_ref = self.model(image_)
                output = self.classifier_head(feature_ref)
                contrastive_loss = contrastive_loss_criterion(feature_ref[:int(image_.shape[0] / 2)], features_pos.detach(),
                                                         features_neg.detach())
                classifier_loss = criterion(output[int(image_.shape[0] / 2):], obj_gt)
                loss = contrastive_loss + self.args.classifier_weight * classifier_loss

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), int(image.size(0) / 2))
            classifier_losses.update(classifier_loss.item(), int(image.size(0) / 2))
            contrastive_losses.update(contrastive_loss.item(), int(image.size(0) / 2))

            pbar.set_description(
                f"[{epoch}/{args.epoch}] loss: {losses.avg:.4f} classifier_loss: {classifier_losses.avg:.4f} "
                f"contrastive_loss: {contrastive_losses.avg:.4f}")
        return {'loss': losses.avg, 'classifier_loss': classifier_losses.avg,
                'contrastive_loss': contrastive_losses.avg}

    def train_erm(self, epoch):
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.rank_loader,dynamic_ncols=True, total=len(self.train_loader))
        for idx, (main_data) in enumerate(pbar):
            self.model.train()
            img, label, _ = main_data['image'], main_data['label'], main_data['index']
            img = img.to(self.device)
            label = label.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.args.amp):
                target_logits = self.model(img)
                label = label.long().reshape(target_logits.shape[0])
                ce_loss = self.criterion(target_logits, label)
                ce_loss = ce_loss.mean()
                loss = ce_loss

            self.optimizer.zero_grad(set_to_none=True)
            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)

            losses.update(loss.item(), img.size(0))

            self._scaler_update()  # activated only when using amp

            pbar.set_description('[{}/{}] ce: {:.3f}'.format(epoch, self.total_epoch, losses.avg, ))

        log_dict = {'ce_loss': losses.avg, }

        return log_dict

    def update_rank(self, epoch):
        for data_dict in self.rank_loader:
            image, target, index = data_dict["image"], data_dict["label"], data_dict['index']
            obj_gt = target
            image = image.to(self.device, non_blocking=True)
            target = obj_gt.to(self.device, non_blocking=True)
            with torch.no_grad():
                logit = self.model(image)
            p = torch.softmax(logit.squeeze(1), dim=1)
            Yg = torch.gather(p, 1, torch.unsqueeze(target, 1))
            old_weights = self.rank_loader.sampler.weights[index]
            new_weights = old_weights - (Yg > self.args.p_critical).long().squeeze(1).cpu()
            self.rank_loader.sampler.weights[index] = torch.clamp(new_weights, 0, 1)
            index.cuda()
            for label_temp in range(self.args.num_classes):
                epoch_str = str(epoch)
                current_idx = self.rank_order[epoch_str][label_temp]
                mask = (target.cuda() == label_temp) & (old_weights.cuda() == 1) & (new_weights.cuda() == 0)
                current_idx.extend(index[mask.cpu()].tolist())
                self.rank_order[epoch_str][label_temp] = current_idx

    def display_images_for_classes(self, rank_order, dataset, cur_epoch, num_images_per_class=10, num_rows=10):
        """
        Display images from a dataset for multiple classes based on rank order indices.

        Args:
            cur_epoch (int) : current epoch number
            rank_order (dict): A dictionary where keys are strings representing classes, and values are lists of indices.
            dataset (Dataset): The dataset object from which images are to be fetched.
            num_images_per_class (int): Number of images to display from each class. Default is 10.
            num_rows (int): Number of images per row in the grid. Default is 10.
        """
        # Loop through each class key and get the images
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
            images_high = torch.stack([dataset.__getitem__(x)['image'] for x in top_ranked][:num_images_per_class])
            images_low = torch.stack([dataset.__getitem__(x)['image'] for x in bottom_ranked][:num_images_per_class])

            combined_tensor = torch.cat((images_high, images_low), dim=0)
            grid = torchvision.utils.make_grid(combined_tensor, nrow=num_rows, normalize=True)
            wandb.log({str(class_id): wandb.Image(grid)})

    def _save_ckpt(self, epoch, name):
        model_net_state = {
            'model': self.model.state_dict(),
            'head': self.classifier_head.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict()
        }

        best_val_acc_ckpt_fpath = os.path.join(self.ckpt_dir, f'{name}.pth')
        torch.save(model_net_state, best_val_acc_ckpt_fpath)


if __name__ == '__main__':
    parser = get_parser()
    args = parse_and_check(parser)
    results = []
    for seed in args.seeds:
        initialize_seeds(seed)
        run = wandb.init(project=args.wandb_project_name, dir='/vol/research/project_storage/',
                         tags=['bar', 'final_run'])
        wandb.run.log_code(".")
        args.seed = seed
        trainer = Trainer(args)
        best_results = trainer.run()
        results.append(best_results)
        run.finish()

    means = {key: np.mean([d[key] for d in results]) for key in results[0].keys()}
    std_devs = {key: np.std([d[key] for d in results]) for key in results[0].keys()}
    max_key_length = max(len(key) for key in means.keys())

    for key in means.keys():
        mean_value = means[key]
        std_dev_value = std_devs[key]
        print(f'avg_{key.ljust(max_key_length)}: {mean_value:.3f}, std: {std_dev_value:.3f}')
