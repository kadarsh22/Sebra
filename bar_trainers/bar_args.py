import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset_dir', default='BAR/dataset0.01.torch', type=str)
    parser.add_argument('--ckpt_dir', default='/vol/research/project_storage/pretrained_models/sebra/bar/', type=str)
    parser.add_argument('--dset_name', type=str, default='BAR')
    parser.add_argument('--method_name', type=str, default='sebra')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--epoch', default=40, type=int)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seeds', nargs='+', default=[1, 564, 234, 843, 9999], help='List of random seed')
    parser.add_argument('--name', type=str)
    parser.add_argument('--num_classes', type=float, default=6)
    parser.add_argument('--lambda_penalty', type=float, default=1.0)
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--wandb_project_name', type=str, default='Self-Paced Multi-shortcut Debiasing')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--log_models', action='store_true', help='whether to visualize data or not')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--beta_inverse', type=float, default=0.7)
    parser.add_argument('--p_critical', type=float, default=0.75)
    parser.add_argument('--gap', type=float, default=2, help="how much rank gap should be added to positive sample")
    parser.add_argument('--temperature', type=float, default=0.15, help="contrastive loss temperature coefficient")
    parser.add_argument('--classifier_weight', type=float, default=0.5, help="weight_of_classifier_weight")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument("--batch_size_stage2", default=256, type=int)

    parser.add_argument("--jtt_up_weight", type=int, default=1)
    parser.add_argument("--bias_id_epoch", type=int, default=1)

    return parser


def parse_and_check(parser, required_args=None):
    args = parser.parse_args()

    if required_args is not None:
        if isinstance(required_args, str):
            required_args = [required_args]
        for a in required_args:
            assert getattr(args, a, None) is not None, f'{a} is required.'

    if getattr(args, 'ckpt_dir', None) is not None:
        assert os.path.isdir(args.ckpt_dir)

    return args
