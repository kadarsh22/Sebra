import os
import torch
import numpy as np
import wandb
import random

from dataset.bar import get_action_dataset_train_test
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm
from utils import add_prefix_dict


class BaseActionTrainer:
    def __init__(self, args):
        self.args = args
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        rank_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        indices = list(range(1958))
        random.shuffle(indices)
        split_point = int(0.9 * len(indices))
        train_index = indices[:split_point]
        valid_index = indices[split_point:]

        train_dataset, valid_dataset, test_dataset, train_no_aug = get_action_dataset_train_test(args, train_transform,
                                                                                                 test_transform,
                                                                                                 rank_transform,
                                                                                                 train_index,
                                                                                                 valid_index)

        self.class_names = train_dataset.class_list
        self.valid_best = 0

        self.train_dataset = train_dataset
        self.train_no_aug = train_no_aug
        self.num_classes = len(self.class_names)
        train_dataset = self._modify_train_set(train_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, pin_memory=True)
        self.rank_loader = DataLoader(train_no_aug, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=False, pin_memory=True)
        self.device = torch.device(0)

        self.total_epoch = args.epoch

        self._setup_models()
        self._setup_criterion()
        self._setup_optimizers()
        name = self._setup_method_name_and_default_name()
        print(name)

        args.ckpt_dir = os.path.join(args.ckpt_dir, name, str(args.seed))
        if not os.path.isdir(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        self.ckpt_dir = args.ckpt_dir

        self.best_val_acc = -1
        self.cond_on_best_val_test_log_dict = {}

    def train(self, epoch):
        raise NotImplementedError

    def _before_train(self):
        raise NotImplementedError

    def _modify_train_set(self, train_dataset):
        return train_dataset

    def _setup_models(self):
        raise NotImplementedError

    def _setup_criterion(self):
        raise NotImplementedError

    def _setup_optimizers(self):
        raise NotImplementedError

    def _setup_method_name_and_default_name(self):
        raise NotImplementedError

    def _save_ckpt(self, epoch, name):
        raise NotImplementedError

    def _loss_backward(self, loss):
        if self.args.amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self, optimizer):
        if self.args.amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def _scaler_update(self):
        if self.args.amp:
            self.scaler.update()

    def _modify_rank_loader(self):
        weights = torch.ones(len(self.train_dataset))
        sampler = WeightedRandomSampler(weights, len(self.train_no_aug), replacement=True)
        self.rank_loader = DataLoader(self.train_no_aug, batch_size=self.args.batch_size,
                                      num_workers=self.args.num_workers,
                                      shuffle=False, pin_memory=self.args.pin_memory, sampler=sampler,
                                      persistent_workers=self.args.num_workers > 0, drop_last=True)

    def eval(self, epoch, loader):
        log_dict = {}
        eval_dict = self.__eval_split(
            epoch, loader, self.args.dset_name)
        log_dict.update(eval_dict)
        return log_dict

    def validate(self, epoch):
        valid_acc_dict = self.eval(epoch, self.valid_loader)
        is_best = False
        valid_acc = valid_acc_dict['accuracy']
        if valid_acc > self.valid_best:
            self.valid_best = valid_acc
            self._save_ckpt(epoch, 'valid_best')
            is_best = True
        return valid_acc_dict, is_best

    @torch.no_grad()
    def __eval_split(self, epoch, loader, dset_name):
        self.model.eval()
        total_label = []
        total_pred = []

        pbar = tqdm(loader, dynamic_ncols=True,
                    desc='[{}/{}] evaluating on biased dataset ({})...'.format(epoch,
                                                                               self.total_epoch,
                                                                               dset_name))
        for data_dict in pbar:
            img = data_dict['image']
            label = data_dict['label']
            img = img.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True)
            feature_ref = self.model(img)
            cls_out = self.classifier_head(feature_ref) ## comment this for computing pd
            if isinstance(cls_out, tuple):
                logits = cls_out[0]
            else:
                logits = cls_out
            pred = logits.argmax(dim=1)
            total_label.append(label)
            total_pred.append(pred)

        total_label = torch.cat(total_label, dim=0).cpu().numpy()
        total_pred = torch.cat(total_pred, dim=0).cpu().numpy()

        acc = np.mean(total_pred == total_label)

        log_dict = {'accuracy': acc}

        return log_dict

    def update_best_and_save_ckpt(self, epoch, log_dict):
        val_acc = log_dict[f'{self.args.dset_name}_accuracy']

        if val_acc <= self.best_val_acc \
                and epoch > 1 and len(self.cond_on_best_val_test_log_dict) > 0:
            log_dict.update(self.cond_on_best_val_test_log_dict)
            return

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            for key, value in log_dict.items():
                if key.startswith(f'{self.args.dset_name}_'):
                    new_key = f'best_{key}'
                    self.cond_on_best_val_test_log_dict[new_key] = value
            self._save_ckpt(epoch, 'best')

        log_dict.update(self.cond_on_best_val_test_log_dict)

    def run(self):
        eval_best_dict = None
        artifact = wandb.Artifact(wandb.run.name, type='model')
        self._before_train()
        for e in range(1, self.args.epoch + 1):
            log_dict = self.train(e)
            val_dict, is_best = self.validate(e)
            val_dict = add_prefix_dict('valid/', val_dict)
            log_dict.update(val_dict)
            eval_dict = self.eval(e, self.test_loader)
            log_dict.update(eval_dict)
            if is_best:
                eval_best_dict = add_prefix_dict('best/', eval_dict)
                log_dict.update(eval_best_dict)
                self._save_ckpt(e, 'best_model')
            wandb.log(log_dict)
        if self.args.log_models:
            artifact.add_file(os.path.join(self.ckpt_dir, 'best_model.pth'))
            wandb.run.log_artifact(artifact)
        return eval_best_dict
