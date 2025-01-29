import os
from abc import abstractmethod
import json
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from common.utils import initialize_seeds
from celeba_trainers.utils import AverageMeterAcc
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from model.classifiers import (
    get_classifier,
    get_transforms,
)
from utils import IdxDataset, add_prefix_dict
from dataset.multiceleba import BiasedCelebA


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.num_classes = 2

        idx2attr = json.load(open("create_datasets/celeba/idx2attr.json", 'r'))
        idx2attr = {int(k): v for k, v in idx2attr.items()}
        dataset2names = {
            'celeba': [idx2attr[args.target_id], idx2attr[args.biasA_id], idx2attr[args.biasB_id]],
        }

        self.target_name = dataset2names[args.dset_name][0]
        self.biasA_name = dataset2names[args.dset_name][1]
        self.biasB_name = dataset2names[args.dset_name][2]
        print(f"Dataset: {args.dset_name}")
        print(f"Target: {self.target_name} Bias: {self.biasA_name} & {self.biasB_name}")

        # exp dir
        self.exp_dir = Path(args.exp_dir) / args.dset_name

        self.cur_epoch = 1
        self.best_epoch = 1
        self.is_best = False
        self.best_val_log_dict = {}
        self.con_test_log_dict = {}

    @property
    @abstractmethod
    def run_name(self):
        pass

    def _setup_all(self):
        args = self.args
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        self.device = torch.device(0)
        self._setup_model()
        self._setup_datasets()
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_wandb()
        self._method_special_setups()

    def _setup_datasets(self):
        args = self.args

        self.train_dataset = self._setup_dataset(is_training=True, split='train')
        self.val_dataset = self._setup_dataset(is_training=False, split='val')
        self.test_dataset = self._setup_dataset(is_training=False, split='test')

        self.train_loader = self._setup_train_loader()
        self.val_loader = DataLoader(
            IdxDataset(self.val_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            IdxDataset(self.test_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True
        )
        print(f"Dataset:"
              f"\n  train dataset size: {len(self.train_dataset)}"
              f"\n    val dataset size: {len(self.val_dataset)}"
              f"\n   test dataset size: {len(self.test_dataset)}")

    def _setup_model(self):
        args = self.args
        classifier = get_classifier(args.arch, num_classes=2, )
        self.model = classifier.cuda()
        print(f"Model:\n  {args.arch}")

    def _setup_models_stage2(self):
        self.classifier = get_classifier(self.args.arch, self.args.num_classes, ).to(self.device)
        self.classifier.fc = torch.nn.Identity()
        for p in self.classifier.fc.parameters():
            p.requires_grad = False
        self.classifier_head = nn.Linear(512, self.args.num_classes).to(self.device)

    def _setup_optimizers_stage2(self):
        parameters = [p for p in self.classifier.parameters() if p.requires_grad] + list(
            self.classifier_head.parameters())
        self.optimizer = torch.optim.SGD(parameters, 3e-4, momentum=0.8, weight_decay=1e-4, )

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def _setup_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), self.args.lr, momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay)

    def _setup_scheduler(self):
        step_size = 10
        if step_size == 0:
            step_size += 1
        self.step_lr = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.65)

    def _setup_wandb(self, name=None):
        if name is None:
            name = self.run_name
        wandb.init(
            project=self.args.wandb_project_name,
            name=name,
            config=self.args,
            tags=['celeba', 'final_run'],
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.run.log_code('.')

    def _setup_dataset(self, is_training=True, split='train'):
        args = self.args
        transform = get_transforms(args.arch, is_training=is_training, dataset=args.dset_name)
        idx2attr = json.load(open("create_datasets/celeba/idx2attr.json", 'r'))
        idx2attr = {int(k): v for k, v in idx2attr.items()}
        target_name = idx2attr[args.target_id]
        biasA_name = idx2attr[args.biasA_id]
        biasB_name = idx2attr[args.biasB_id]
        dataset = BiasedCelebA(root=args.root, target_name=target_name, biasA_name=biasA_name, biasB_name=biasB_name,
                               biasA_ratio=args.biasA_ratio, biasB_ratio=args.biasB_ratio, split=split,
                               transform=transform)
        return dataset

    def _setup_train_loader(self):
        args = self.args
        weights = torch.ones(len(self.train_dataset))  # my weights
        sampler = WeightedRandomSampler(weights, len(self.train_dataset), replacement=True)
        train_loader = DataLoader(IdxDataset(self.train_dataset), batch_size=args.batch_size, shuffle=False,
                                  sampler=sampler, num_workers=args.num_workers, pin_memory=True)
        return train_loader

    def _save_checkpoint(self, name=None, model=None, seed=None):
        if model is None:
            model = self.model
        if name is None:
            name = "last"
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.cur_epoch,
        }
        ckpt_fpath = os.path.join(self.exp_dir, str(seed))
        os.makedirs(ckpt_fpath, exist_ok=True)
        torch.save(ckpt, Path(ckpt_fpath) / f'{name}.pth')

    def _method_special_setups(self):
        pass

    def _before_train(self):
        pass

    def _after_train(self):
        pass

    def _batch_transform(
            self,
            batch_data,
            batch_label,
            # batch_indices
    ):
        """
        Transform batch data in training
        """
        return batch_data, batch_label

    def _loss_backward(self, loss, retain_graph=False):
        if self.args.amp:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def _optimizer_step(self, optimizer):
        if self.args.amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def _scaler_update(self):
        if self.args.amp:
            self.scaler.update()

    @abstractmethod
    def train(self):
        pass

    @torch.no_grad()
    def infer_bias_label(self, model=None) -> torch.LongTensor:
        """training set: Correct as bias-aligned (bias=1), error as bias-conflicted (bias=0)"""
        args = self.args
        # get bias label
        bias_label_list = []

        if model is None:
            model = self.model
        model.eval()

        unshuffled_train_loader = DataLoader(
            IdxDataset(self.train_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
        )

        group_num_list = [{'total': 0, 'error': 0} for _ in range(self.train_dataset.group_num)]
        for indices, (image, target) in tqdm(unshuffled_train_loader, leave=False):
            image = image.cuda()
            with torch.cuda.amp.autocast(enabled=args.amp):
                output, _ = model(image)

            pred = output.argmax(dim=1).cpu()
            label = target[:, 0]

            bias_label = (pred == label).long()
            bias_label_list.append(bias_label)

            error = (pred != label).bool()
            batch_group_label = self.train_dataset.group_label[indices]
            for i in range(self.train_dataset.group_num):
                group_indices = torch.where(batch_group_label == i)[0]
                group_num_list[i]['total'] += len(group_indices)
                group_num_list[i]['error'] += torch.sum(error[group_indices])

        for i, dic in enumerate(group_num_list):
            print(f"group{i}: {dic['error']}/{dic['total']} ", end='')
            if i == 3 or i == 7: print("")

        bias_label = torch.concat(bias_label_list, dim=0)
        return bias_label

    def eval(self):
        args = self.args
        val_log_dict = self._eval_split('val')
        test_log_dict = self._eval_split('test')

        print("***************************** group acc ************************************")
        print("val:  ", end="")
        for i in range(self.val_dataset.group_num):
            if i == 4:
                print("      ", end="")
            print(f"group{i} acc {val_log_dict[f'val_group_{i}_acc']:.1f}", end=" ")
            if i == 3 or i == 7:
                print("")
        print("test: ", end="")
        for i in range(self.val_dataset.group_num):
            if i == 4:
                print("      ", end="")
            print(f"group{i} acc {test_log_dict[f'test_group_{i}_acc']:.1f}", end=" ")
            if i == 3 or i == 7:
                print("")
        print("**************************** test metrics *********************************")
        print("current test")
        print(f"id acc {test_log_dict['test_id_acc']:.1f} || "
              f"group avg acc {test_log_dict['test_group_avg_acc']:.1f} || "
              f"worst group acc {test_log_dict['test_worst_group_acc']:.1f} || "
              f"{self.biasA_name} gap {test_log_dict[f'test_{self.biasA_name}_gap']:.1f} || "
              f"{self.biasB_name} gap {test_log_dict[f'test_{self.biasB_name}_gap']:.1f}")

        # check whether is best
        compare_key = f"val_{args.model_selection_meter}"
        for key, value in val_log_dict.items():
            new_key = f"best_{key}"
            if (
                    new_key not in self.best_val_log_dict.keys() or
                    value >= self.best_val_log_dict[new_key]
            ):
                self.best_val_log_dict[new_key] = value
                if key == compare_key:
                    self.is_best = True
                    self.con_test_log_dict = {
                        f"con_{k}": v for k, v in test_log_dict.items()
                    }
                    self.best_epoch = self.cur_epoch

        print(f"Best epoch: {self.best_epoch}    (by {args.model_selection_meter})")
        con_test_log_dict = self.con_test_log_dict
        print(f"id acc {con_test_log_dict['con_test_id_acc']:.1f} || "
              f"group avg acc {con_test_log_dict['con_test_group_avg_acc']:.1f} || "
              f"worst group acc {con_test_log_dict['con_test_worst_group_acc']:.1f} || "
              f"{self.biasA_name} gap {con_test_log_dict[f'con_test_{self.biasA_name}_gap']:.1f} || "
              f"{self.biasB_name} gap {con_test_log_dict[f'con_test_{self.biasB_name}_gap']:.1f}")
        print("****************************************************************************\n")

        # add custom step
        val_log_dict.update({
            "custom_step": self.cur_epoch
        })
        test_log_dict.update({
            "custom_step": self.cur_epoch
        })
        self.best_val_log_dict.update({
            "custom_step": self.cur_epoch
        })
        self.con_test_log_dict.update({
            "custom_step": self.cur_epoch
        })

        val_log_dict = add_prefix_dict('val/', val_log_dict)
        test_log_dict = add_prefix_dict('test/', test_log_dict)
        best_val_log_dict = add_prefix_dict('best_val/', self.best_val_log_dict)
        con_test_log_dict = add_prefix_dict('best_test/', con_test_log_dict)
        self.log_to_wandb(val_log_dict)
        self.log_to_wandb(test_log_dict)
        self.log_to_wandb(best_val_log_dict)
        self.log_to_wandb(con_test_log_dict)
        return self.is_best, con_test_log_dict

    @torch.no_grad()
    def _eval_split(self, split='val'):
        args = self.args

        if split == 'val':
            loader = self.val_loader
        else:
            loader = self.test_loader

        accuracy_dict = {
            f"group_{i}": AverageMeterAcc() for i in range(self.val_dataset.group_num)
        }
        accuracy_dict.update({
            "normal": AverageMeterAcc(),
            "biasA": AverageMeterAcc(),
            "biasB": AverageMeterAcc()
        })

        self.model.eval()
        pbar = tqdm(loader, dynamic_ncols=True, leave=False)
        for indices, data_dict in pbar:
            image = data_dict['image'].to(self.device)
            target = data_dict['label']
            bs = len(image)

            with torch.cuda.amp.autocast(enabled=args.amp):
                feature_ref = self.classifier(image)
                output = self.classifier_head(feature_ref)

            pred = output.argmax(dim=1).cpu()

            task_label = target[:, 0]
            biasA_label = target[:, 1]
            biasB_label = target[:, 2]

            task_correct = (pred == task_label).bool()
            biasA_correct = (pred == biasA_label).bool()
            biasB_correct = (pred == biasB_label).bool()

            accuracy_dict["normal"].update(
                correct_num=torch.sum(task_correct).item(),
                batch_size=bs
            )
            accuracy_dict["biasA"].update(
                correct_num=torch.sum(biasA_correct).item(),
                batch_size=bs
            )
            accuracy_dict["biasB"].update(
                correct_num=torch.sum(biasB_correct).item(),
                batch_size=bs
            )

            batch_group_label = loader.dataset.dataset.group_label[indices]
            for i in range(self.val_dataset.group_num):
                group_indices = torch.where(batch_group_label == i)[0]
                if len(group_indices) > 0:
                    accuracy_dict[f"group_{i}"].update(
                        correct_num=torch.sum(task_correct[group_indices]),
                        batch_size=len(group_indices)
                    )

        log_dict = self._build_log_dict(
            split=split,
            accuracy_dict=accuracy_dict
        )
        return log_dict

    def _build_log_dict(self, split, accuracy_dict):
        args = self.args
        log_dict = {
            f"{split}_{k}_acc": v.avg for k, v in accuracy_dict.items()
        }

        group_acc_list = [accuracy_dict[f"group_{i}"].avg for i in range(self.val_dataset.group_num)]

        # common uncommon group
        log_dict.update({
            f"{split}_common_{self.biasA_name}_common_{self.biasB_name}_acc":
                (group_acc_list[0] + group_acc_list[4]) / 2,
            f"{split}_common_{self.biasA_name}_uncommon_{self.biasB_name}_acc":
                (group_acc_list[1] + group_acc_list[5]) / 2,
            f"{split}_uncommon_{self.biasA_name}_common_{self.biasB_name}_acc":
                (group_acc_list[2] + group_acc_list[6]) / 2,
            f"{split}_uncommon_{self.biasA_name}_uncommon_{self.biasB_name}_acc":
                (group_acc_list[3] + group_acc_list[7]) / 2,
        })

        worst_group_acc = min(log_dict[f"{split}_common_{self.biasA_name}_common_{self.biasB_name}_acc"],
                              log_dict[f"{split}_common_{self.biasA_name}_uncommon_{self.biasB_name}_acc"],
                              log_dict[f"{split}_uncommon_{self.biasA_name}_common_{self.biasB_name}_acc"],
                              log_dict[f"{split}_uncommon_{self.biasA_name}_uncommon_{self.biasB_name}_acc"])

        log_dict.update({'worst_subgroup_acc': worst_group_acc})

        # i.d. acc
        group_ratio_list = [
            args.biasA_ratio * args.biasB_ratio * 0.5,
            args.biasA_ratio * (1 - args.biasB_ratio) * 0.5,
            (1 - args.biasA_ratio) * args.biasB_ratio * 0.5,
            (1 - args.biasA_ratio) * (1 - args.biasB_ratio) * 0.5,
            args.biasA_ratio * args.biasB_ratio * 0.5,
            args.biasA_ratio * (1 - args.biasB_ratio) * 0.5,
            (1 - args.biasA_ratio) * args.biasB_ratio * 0.5,
            (1 - args.biasA_ratio) * (1 - args.biasB_ratio) * 0.5
        ]
        id_acc = np.sum(np.multiply(group_acc_list, group_ratio_list))
        log_dict.update({
            f"{split}_id_acc": id_acc
        })

        # group avg acc
        group_avg_acc = np.mean(group_acc_list)
        log_dict.update({
            f"{split}_group_avg_acc": group_avg_acc
        })

        # worst group acc
        worst_group_acc = np.min(group_acc_list)
        log_dict.update({
            f"{split}_worst_group_acc": worst_group_acc
        })

        # bias gap
        biasA_gap = (
                log_dict[f"{split}_uncommon_{self.biasA_name}_common_{self.biasB_name}_acc"]
                - id_acc
        )
        biasB_gap = (
                log_dict[f"{split}_common_{self.biasA_name}_uncommon_{self.biasB_name}_acc"]
                - id_acc
        )
        both_gap = (
                log_dict[f"{split}_uncommon_{self.biasA_name}_uncommon_{self.biasB_name}_acc"]
                - id_acc
        )
        log_dict.update({
            f"{split}_{self.biasA_name}_gap": biasA_gap,
            f"{split}_{self.biasB_name}_gap": biasB_gap,
            f"{split}_both_gap": both_gap,
        })

        for key in log_dict.keys():
            log_dict[key] *= 100

        return log_dict

    def visualise_predictions(self, img_size=28, plot_name='predictions'):
        data = [(images, torch.max(self.model(images.cuda().view(-1, 3, img_size, img_size)).data, 1)[1]) for
                _, (images, attr) in self.test_loader][:-2]
        x = torch.stack([d[0] for d in data]).view(-1, 3, img_size, img_size)
        l = torch.stack([d[1] for d in data]).view(-1)
        images = []
        for i in range(2):
            if x[l.cpu() == i][:10].shape[0] == 10:
                images.append(x[l.cpu() == i][:10])
            else:
                images.append(torch.zeros((10, 3, img_size, img_size)))
        images = torch.stack(images).view(-1, 3, img_size, img_size)
        grid_img = torchvision.utils.make_grid(images[:20], nrow=10, normalize=False)
        plt.imshow(grid_img.permute(1, 2, 0).cpu().data)
        wandb.log({plot_name: wandb.Image(grid_img)})

    def log_to_wandb(self, log_dict):
        assert "custom_step" in log_dict.keys()
        if self.args.wandb:
            wandb.log(log_dict)

    def __call__(self):
        args = self.args

        best_results = None
        results_avg = []
        for seed in args.seeds:
            self.cur_epoch = 0
            initialize_seeds(seed)
            self.args.seed = seed
            self.is_best = False
            self.best_val_log_dict = {}
            self.con_test_log_dict = {}
            self._setup_all()
            artifact = wandb.Artifact('temp', type='model')
            self._before_train()
            for e in range(1, args.epoch + 1):
                self.train()
                self._after_train()
                if args.not_eval is False:
                    is_best, best_test_log_dict = self.eval()
                if args.save_ckpt and is_best:
                    self._save_checkpoint('best_model_bb', self.classifier, seed=seed)
                    self._save_checkpoint('best_model_head', self.classifier_head, seed=seed)
                self.cur_epoch += 1

            results_avg.append(best_test_log_dict)
            if self.args.log_models:
                artifact.add_file(os.path.join(self.exp_dir, str(seed), 'best_model_bb.pth'))
                artifact.add_file(os.path.join(self.exp_dir, str(seed), 'best_model_head.pth'))
                wandb.run.log_artifact(artifact)
            wandb.run.finish()

        means = {key: np.mean([d[key] for d in results_avg]) for key in results_avg[0].keys()}
        std_devs = {key: np.std([d[key] for d in results_avg]) for key in results_avg[0].keys()}
        max_key_length = max(len(key) for key in means.keys())
        
        for key in means.keys():
            mean_value = means[key]
            std_dev_value = std_devs[key]
            print(f'avg_{key.ljust(max_key_length)}: {mean_value:.3f}, std: {std_dev_value:.3f}')
