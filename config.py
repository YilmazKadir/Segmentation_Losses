import argparse


def str2opt(arg):
  assert arg in ['SGD', 'Adam', 'AdamW']
  return arg


def str2scheduler(arg):
  assert arg in ['StepLR', 'PolyLR', 'ExpLR', 'SquaredLR', 'OneCycleLR']
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


def str2list(l):
  return [int(i) for i in l.split(',')]


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


arg_lists = []
parser = argparse.ArgumentParser()

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='UNet', help='Model name')
net_arg.add_argument(
    '--conv1_kernel_size', type=int, default=5, help='First layer conv kernel size')
net_arg.add_argument('--weights', type=str, default='None', help='Saved weights to load')
net_arg.add_argument(
    '--dilations', type=str2list, default='1,1,1,1', help='Dilations used for ResNet or DenseNet')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='AdamW')
opt_arg.add_argument('--lr', type=float, default=1e-2)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--param_histogram_freq', type=int, default=100)
opt_arg.add_argument('--save_param_histogram', type=str2bool, default=False)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.02)

# Scheduler
opt_arg.add_argument('--scheduler', type=str2scheduler, default='OneCycleLR')
opt_arg.add_argument('--max_iter', type=int, default=90000)
opt_arg.add_argument('--step_size', type=int, default=2e4)
opt_arg.add_argument('--step_gamma', type=float, default=0.1)
opt_arg.add_argument('--poly_power', type=float, default=0.9)
opt_arg.add_argument('--exp_gamma', type=float, default=0.95)
opt_arg.add_argument('--exp_step_size', type=float, default=445)

# Directories
dir_arg = add_argument_group('Directories')
dir_arg.add_argument('--log_dir', type=str, default='outputs/default')
dir_arg.add_argument('--data_dir', type=str, default='data')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ADE20KDataset')
# StanfordArea5Dataset ScannetVoxelizationDataset KITTIVoxelizationDataset
data_arg.add_argument('--temporal_dilation', type=int, default=10)
data_arg.add_argument('--temporal_numseq', type=int, default=3)
data_arg.add_argument('--temporal_rand_dilation', type=str2bool, default=False)
data_arg.add_argument('--temporal_rand_numseq', type=str2bool, default=False)
data_arg.add_argument('--point_lim', type=int, default=-1)
data_arg.add_argument('--batch_size', type=int, default=2)
data_arg.add_argument('--val_batch_size', type=int, default=2)
data_arg.add_argument('--test_batch_size', type=int, default=2)
data_arg.add_argument(
    '--num_workers', type=int, default=4, help='num workers for train/test dataloader')
data_arg.add_argument('--num_val_workers', type=int, default=4, help='num workers for val dataloader')
data_arg.add_argument('--ignore_mask', type=int, default=255)
data_arg.add_argument('--train_limit_numpoints', type=int, default=0)
data_arg.add_argument('--merge', type=str2bool, default=False)

data_arg.add_argument(
    '--ade20k_path',
    type=str,
    default='/work/scratch/kyilmaz/datasets/ADE20K',
    help='ADE20K dataset root dir')

data_arg.add_argument(
    '--scannet_path',
    type=str,
    default='/globalwork/yilmaz/data/processed/minkowski_scannet',
    help='Scannet online voxelization dataset root dir')

data_arg.add_argument(
    '--stanford3d_path',
    type=str,
    default='/globalwork/yilmaz/data/processed/minkowski_S3DIS',
    help='Stanford precropped dataset root dir')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--stat_freq', type=int, default=40, help='print frequency')
train_arg.add_argument('--test_stat_freq', type=int, default=100, help='print frequency')
train_arg.add_argument('--save_freq', type=int, default=1000, help='save frequency')
train_arg.add_argument('--val_freq', type=int, default=1000, help='validation frequency')
train_arg.add_argument(
    '--empty_cache_freq', type=int, default=1, help='Clear pytorch cache frequency')
train_arg.add_argument('--train_phase', type=str, default='train', help='Dataset for training')
train_arg.add_argument('--val_phase', type=str, default='validation', help='Dataset for validation')
train_arg.add_argument(
    '--overwrite_weights', type=str2bool, default=True, help='Overwrite checkpoint during training')
train_arg.add_argument(
    '--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
train_arg.add_argument(
    '--resume_optimizer',
    default=True,
    type=str2bool,
    help='Use checkpoint optimizer states when resume training')

# Data augmentation
data_aug_arg = add_argument_group('DataAugmentation')
data_aug_arg.add_argument('--normalize_color', type=str2bool, default=True)

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--visualize', type=str2bool, default=False)
test_arg.add_argument('--test_temporal_average', type=str2bool, default=False)
test_arg.add_argument('--visualize_path', type=str, default='outputs/visualize')
test_arg.add_argument('--save_prediction', type=str2bool, default=False)
test_arg.add_argument('--save_pred_dir', type=str, default='outputs/pred')
test_arg.add_argument('--test_phase', type=str, default='test', help='Dataset for test')
test_arg.add_argument('--test_original_pc', type=str2bool, default=True, help='Test on the original pointcloud space.')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True)


def get_config():
  config = parser.parse_args()
  return config  # Training settings
  
