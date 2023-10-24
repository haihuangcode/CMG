import argparse

parser = argparse.ArgumentParser(description="A project implemented in pyTorch")

# =========================== Learning Configs ============================
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--n_epoch', type=int)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('--test_batch_size', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--gpu', type=str)
parser.add_argument('--snapshot_pref', type=str)
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--clip_gradient', type=float)
parser.add_argument('--loss_weights', type=float)
parser.add_argument('--start_epoch', type=int)
parser.add_argument('--model_save_path', default='../../../checkpoint')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
parser.add_argument('--weight_decay', '--wd', type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

# =========================== Display Configs ============================
parser.add_argument('--print_freq', type=int)
parser.add_argument('--save_freq', type=int)
parser.add_argument('--eval_freq', type=int)