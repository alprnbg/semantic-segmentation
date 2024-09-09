import os
import glob
import shutil
import argparse
import albumentations


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wandb_username", default=None,
                        help="Wandb username")
    parser.add_argument("--wandb_project_name",
                        default=None, help="Wandb project name")

    parser.add_argument("--optuna_search_config", default=None, help="Path of the YAML file containing information for"
                                                                     "hyperparam search with Optuna.", type=str)
    parser.add_argument("--search_num_trials", default=None,
                        help="# of trials that optuna will run", type=int)
    parser.add_argument("--search_timeout", default=None,
                        help="timeout for optuna search (see optuna docs)", type=int)

    parser.add_argument("--test_dataset_path", default=None, help="Path of the test dataset, testing is disabled if not"
                                                                  "supplied", type=str)

    parser.add_argument("--test_n_epochs",
                        default=1,
                        help="Performs evaluation in test set every specified epochs, saves the model and test results",
                        type=int)

    parser.add_argument('--train_val_ratio', type=float, default=0.8,
                        help="Specify the portion of the dataset that will be used for training in train-val split")

    parser.add_argument('--draw-val', action="store_true", help="If True, draws predictions and gts for validation set just for the last epoch.")
    parser.add_argument('--channels-last', action="store_true", default=False,
                        help="Uses channels last memory format for faster training and lower memory usage on "
                             "GPU's with Tensor cores")

    # =========================================

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 4)')

    # model
    parser.add_argument('--arch', '-a', default='Unet3Convnext4xDS')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--weight', default=None, type=str, help="weight checkpoint")

    # loss
    parser.add_argument('--loss', '-l', default='BCEDiceLoss')

    # optimizer
    parser.add_argument('--optimizer', '-o', default='Radam')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0)

    # scheduler
    parser.add_argument('--scheduler', '-s', default='ConstantLR')
    parser.add_argument('--min_lr', default=1e-5)
    parser.add_argument('--factor', default=None)
    parser.add_argument('--patience', default=None)
    parser.add_argument('--milestones', default=None)
    parser.add_argument('--gamma', default=None)

    # dataset
    parser.add_argument('--dataset', default='sample_dataset',
                        help='dataset name')
    parser.add_argument('--subset_file', default=None,
                        help='if given, uses images given in the file only in the training')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    parser.add_argument('--input_size_w', default=None, type=int,
                        help="Input width. If none no resize or cropping is applied")
    parser.add_argument('--input_size_h', default=None, type=int,
                        help="Input height. If none no resize or cropping is applied")
    parser.add_argument('--resize_method', default=None,
                        help="If data is already tiled (ie. input_size_w or input_size_h is given None), this is not applied."
                             "options: Resize, RandomCrop, RandomResizedCrop, Padding. see albumentation")

    parser.add_argument("--val_partition", action="store_true", 
                        help="If enabled, validation is done via partitioning "
                             "(resize is not applied to val set in this case)")

    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=1, type=int)

    config = parser.parse_args()

    return config


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


inv_normalize = albumentations.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255],
    max_pixel_value=1.)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def record_code_files(log_folder):
    py_files = glob.glob(os.path.join("**","*.py"), recursive=True)
    for f in py_files:
        if f.split(os.sep)[0] == "models":
            continue
        folder_path = os.path.join(log_folder, (os.sep).join(f.split(os.sep)[:-1]))
        os.makedirs(folder_path ,exist_ok=True)
        shutil.copy2(f, folder_path)


def suggest_config(config, optuna_config, trial):
    for attr, search in optuna_config.items():
        search_type = search[0]
        values = search[1]
        assert search_type in ["categorical", "discrete_uniform",
                               "float", "int", "loguniform",
                               "uniform"]
        if search_type == "categorical":
            config[attr] = trial.suggest_categorical(attr, values)
        elif search_type == "discrete_uniform":
            assert len(values) == 3
            config[attr] = trial.suggest_discrete_uniform(attr, *values)
        elif search_type == "float":
            assert len(values) == 2
            config[attr] = trial.suggest_float(attr, *values)
        elif search_type == "int":
            assert len(values) == 2
            config[attr] = trial.suggest_int(attr, *values)
        elif search_type == "loguniform":
            assert len(values) == 2
            config[attr] = trial.suggest_loguniform(attr, *values)
        elif search_type == "uniform":
            assert len(values) == 2
            config[attr] = trial.suggest_uniform(attr, *values)
    return config


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
