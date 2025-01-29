import os
import torch
import wandb
import pickle
import numpy as np
from glob import glob
from common.utils import initialize_seeds
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AverageMeter
from bar_args import get_parser, parse_and_check
from bar_trainers.base_trainer import BaseActionTrainer
from model.model_zoo import get_model_and_transforms


class PDTrainer(BaseActionTrainer):
    def __init__(self, args):
        super(PDTrainer, self).__init__(args)

    def _setup_models(self, pretrained=True):
        self.model, _ = get_model_and_transforms(args.arch, num_classes=6, pretrained=pretrained)

    def _setup_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def _setup_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _setup_method_name_and_default_name(self):
        self.args.method_name = 'sebra_'
        default_name = 'bar_'
        self.default_name = default_name

    def train(self, epoch):
        losses = AverageMeter("Loss", ":.4e")

        pbar = tqdm(self.train_loader,
                    dynamic_ncols=True, total=len(self.train_loader))
        for idx, (main_data) in enumerate(pbar):
            # ============= start: train classifier net ================
            self.model.train()
            img, label, _ = main_data
            img = main_data['image']
            label = main_data['label']
            img = img.to(self.device)
            label = label.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.args.amp):
                target_logits = self.model(img)

                label = label.long()
                label = label.reshape(target_logits.shape[0])

                # standard CE or BCE loss
                ce_loss = self.criterion(target_logits, label)
                ce_loss = ce_loss.mean()
                loss = ce_loss

            self.optimizer.zero_grad(set_to_none=True)
            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)

            losses.update(loss.item(), img.size(0))
            # ============= end: train classifier net ================

            self._scaler_update()  # activated only when using amp

            # pbar.set_description('[{}/{}] ce: {:.3f}'.format(epoch, self.total_epoch, losses.avg, ))

        log_dict = {'ce_loss': losses.avg, }

        return log_dict

    def _get_spurious_ranking(self, path):
        with open(path, 'rb') as f:
            spurious_ranking = pickle.load(f)
        ranking_compiled = {}
        for i in range(6):
            class_wise_rank_order = []
            for rank, rank_dict in spurious_ranking.items():
                class_wise_rank_order.extend(list(OrderedDict.fromkeys(rank_dict[i])))
            ranking_compiled[i] = class_wise_rank_order
        return ranking_compiled

    def compute_pd(self):
        args = self.args
        paths = ['/user/HS502/ak03476/PycharmProjects/sebra/exp/bar_erm_order.pkl']
        paths = ['assets/ranking/bar/neurips23mmoayeri/index.pkl'] ## spuriosity_ranking
        # paths = ['exp/ranks_bar/0.7_0.9_0.001_0.001_128_rank_order_bar_ours.pkl'] ## sebra
        self._setup_models(pretrained=True)
        self._setup_optimizers()
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()
        for i, path in enumerate(paths):
            self._setup_models()
            self._setup_criterion()
            self._setup_optimizers()
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            if args.method_name == 'sebra':
                ranking = self._get_spurious_ranking(path)
                top_ranks_idx, bot_ranks_idx = [] , []
                for _, idx in ranking.items():
                    top_ranks_idx.append(idx[:100])
                for _, idx in ranking.items():
                    bot_ranks_idx.append(idx[-100:])
            else:
                with open(path, 'rb') as f:
                    spurious_ranking = pickle.load(f)
                top_ranks = spurious_ranking['top_idx']['train']
                bot_ranks = spurious_ranking['bot_idx']['train']
                top_ranks_idx, bot_ranks_idx = [], []
                for _, idx in top_ranks.items():
                    top_ranks_idx.append(idx[:200])
                for _, idx in bot_ranks.items():
                    bot_ranks_idx.append(idx[-200:])
            high_ranks_idx = torch.LongTensor(np.concatenate(top_ranks_idx))
            high_spurious_data = torch.utils.data.Subset(self.train_dataset, high_ranks_idx)
            self.train_loader = DataLoader(high_spurious_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                           shuffle=True, pin_memory=True, drop_last=False)
            accs_high_spurious = []
            for e in range(1, args.epoch + 1):
                log_dict = self.train(e)
                eval_dict = self.eval(e, self.test_loader)
                log_dict.update(eval_dict)
                accs_high_spurious.append(log_dict["accuracy"])
                wandb.log(log_dict)

            self._setup_models()
            self._setup_criterion()
            self._setup_optimizers()
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)

            low_ranks_idx = torch.LongTensor(np.concatenate(bot_ranks_idx))
            low_spurious_data = torch.utils.data.Subset(self.train_dataset, low_ranks_idx)
            self.train_loader = DataLoader(low_spurious_data, batch_size=args.batch_size, num_workers=args.num_workers,
                                           shuffle=True, pin_memory=True, drop_last=False)
            accs_low_spurious = []
            for e in range(1, args.epoch + 1):
                log_dict = self.train(e)
                eval_dict = self.eval(e, self.test_loader)
                log_dict.update(eval_dict)
                accs_low_spurious.append(log_dict["accuracy"])
                wandb.log(log_dict)
            pd = max(accs_low_spurious) - max(accs_high_spurious)
            print(f"Performance Disparity metric: {pd}")

            return pd


    def _save_ckpt(self, epoch, name):
        model_net_state = {
            'model': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict()
        }

        best_val_acc_ckpt_fpath = os.path.join(self.ckpt_dir, 'best.pth')
        torch.save(model_net_state, best_val_acc_ckpt_fpath)


if __name__ == '__main__':
    parser = get_parser()
    args = parse_and_check(parser)
    performance_disparity_accumulator = []
    for seed in args.seeds:
        args.seed = seed
        args.method_name = 'spuriosity_ranking'
        initialize_seeds(args.seed)
        run = wandb.init(project=args.wandb_project_name, dir='/vol/research/project_storage/',
                         tags=['bar', 'eval'],name='bar_pd_eval')
        wandb.run.log_code(".")
        trainer = PDTrainer(args)
        performance_disparity = trainer.compute_pd()
        performance_disparity_accumulator.append(performance_disparity)

    mean = np.mean(performance_disparity_accumulator)
    std_dev = np.std(performance_disparity_accumulator)

    print(f"Mean: {mean}, Standard Deviation: {std_dev}")

