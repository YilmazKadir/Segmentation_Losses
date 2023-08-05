import argparse


def str2bool(v):
  return v.lower() in ('true', '1')

def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg

arg_lists = []
parser = argparse.ArgumentParser()

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='vit', help='Model name')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--lr', type=float, default=1e-2)

# Scheduler
opt_arg.add_argument('--max_epochs', type=int, default=50)

# Directories
dir_arg = add_argument_group('Directories')
dir_arg.add_argument('--log_dir', type=str, default='outputs/default')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='Cityscapes')
data_arg.add_argument('--batch_size', type=int, default=2)
data_arg.add_argument('--val_batch_size', type=int, default=2)
data_arg.add_argument('--test_batch_size', type=int, default=2)
data_arg.add_argument('--num_workers', type=int, default=4)
data_arg.add_argument('--num_val_workers', type=int, default=4)
data_arg.add_argument('--ignore_index', type=int, default=255)
data_arg.add_argument('--cityscapes_path',type=str,default='/images/PublicDataset/cityscapes')
data_arg.add_argument('--synapse_path',type=str,default='/home/students/kyilmaz/project_TransUNet/data/Synapse')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--stat_freq', type=int, default=40, help='print frequency')
train_arg.add_argument('--save_freq', type=int, default=1000, help='save frequency')
train_arg.add_argument('--val_freq', type=int, default=1000, help='validation frequency')
train_arg.add_argument('--train_phase', type=str, default='train', help='Dataset for training')
train_arg.add_argument('--val_phase', type=str, default='validation', help='Dataset for validation')
train_arg.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')

# Test
test_arg = add_argument_group('Test')
test_arg.add_argument('--test_phase', type=str, default='test', help='Dataset for test')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_cuda', type=str2bool, default=True)


def get_config():
  config = parser.parse_args()
  return config  # Training settings
  
