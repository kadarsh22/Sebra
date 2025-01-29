"""
# --------------------------------------------------------
# implementation from Echoes:
# https://github.com/isruihu/Echoes
# --------------------------------------------------------
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import argparse
from celeba_trainers import *



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default='celeba')
    parser.add_argument("--method", required=True)
    parser.add_argument("--arch", type=str, default='resnet18')
    parser.add_argument("--root", type=str, default="data/")
    parser.add_argument("--exp_dir", type=str, default="/vol/research/project_storage/pretrained_models/sebra/")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--seeds', nargs='+', default=[1, 843, 999, 245, 392], help='List of seeds')
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument('--project', type=str, default='Self-Paced Multi-shortcut Debiasing')
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--log_models", action="store_true")
    parser.add_argument("--not_eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_offline", action="store_true")

    # dataset attributes
    parser.add_argument("--target_id", type=int, default=31)
    parser.add_argument("--biasA_id", type=int, default=20)
    parser.add_argument("--biasB_id", type=int, default=39)
    parser.add_argument("--biasA_ratio", type=float, default=0.95)
    parser.add_argument("--biasB_ratio", type=float, default=0.95)
    parser.add_argument("--num_classes", type=int, default=2)


    parser.add_argument("--model_selection_meter", type=str,
                        choices=['worst_group_acc', 'normal_acc', 'id_acc'], default='worst_group_acc')

    parser.add_argument("--group_label", type=str, default='both')

    # JTT
    parser.add_argument("--bias_epoch", type=int, default=1)
    parser.add_argument("--jtt_up_weight", type=int, default=50)
    parser.add_argument("--bias_label_type", type=str, choices=["biasA", "biasB", "auto", ], default="auto")
    # EIIL
    parser.add_argument("--eiil_n_steps", type=int, default=10000)

    # GroupDRO
    parser.add_argument("--groupdro_robust_step_size", type=float, default=0.01)
    parser.add_argument("--groupdro_gamma", type=float, default=0.1)

    # Sebra
    parser.add_argument("--epoch", type=int, default=30 )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_stage2", default=64, type=int)
    parser.add_argument('--beta_inverse', type=float, default=0.8)
    parser.add_argument('--p_critical', type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=0.05, help="contrastive loss temperature coefficient")
    parser.add_argument('--gap', type=float, default=2, help="how much rank gap should be added to positive sample")
    parser.add_argument('--classifier_weight', type=float, default=0.5, help="weight_of_classifier_weight")

    args = parser.parse_args()

    if args.wandb:
        args.wandb_project_name = args.project
        if args.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'

    return args


if __name__ == '__main__':
    args = parse_args()
    method = methods[args.method](args)
    print("Method: {}".format(method.method_name))
    method()
