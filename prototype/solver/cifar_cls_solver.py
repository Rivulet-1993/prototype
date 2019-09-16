import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time
import datetime
import torch
import linklink as link
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

from .base_solver import BaseSolver
from .cls_solver import ClsSolver
from prototype.config import parse_config
from prototype.utils.dist import link_dist, DistModule
from prototype.utils.misc import makedir, create_logger, get_logger, count_params, count_flops, \
    param_group_all, AverageMeter, accuracy, load_state_model, load_state_optimizer
from prototype.utils.ema import EMA
from prototype.model import model_entry
from prototype.optimizer import optim_entry, FP16RMSprop, FP16SGD, FusedFP16SGD
from prototype.lr_scheduler import scheduler_entry
from prototype.data import make_imagenet_train_data, make_imagenet_val_data
from prototype.loss_functions import LabelSmoothCELoss
from prototype.data.sampler import DistributedGivenIterationSampler, DistributedEpochSampler, DistributedSampler
# from prototype.data.nvidia_dali_dataloader import DaliDataLoader, dali_default_collate
from prototype.data.autoaugment import CIFAR10Policy, Cutout


class CifarClsSolver(ClsSolver):

    def build_data(self):
        self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        self.config.data.last_iter = self.state['last_iter']

        if self.config.data.task == 'cifar10':
            assert self.config.model.kwargs.num_classes == 10

            aug = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if self.config.data.get('autoaugment', False):
                aug.append(CIFAR10Policy())

            aug.append(transforms.ToTensor())

            if self.config.data.get('cutout', False):
                aug.append(Cutout(n_holes=1, length=16))

            aug.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            )
            transform_train = transforms.Compose(aug)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_dataset = CIFAR10(root='/mnt/lustre/yuankun/transfer_datasets/cifar10/', 
                                    train=True, download=False, transform=transform_train)
            val_dataset = CIFAR10(root='/mnt/lustre/yuankun/transfer_datasets/cifar10/', 
                                    train=False, download=False, transform=transform_test)

        elif self.config.data.task == 'cifar100':
            assert self.config.model.kwargs.num_classes == 100

            aug = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if self.config.data.get('autoaugment', False):
                aug.append(CIFAR10Policy())

            aug.append(transforms.ToTensor())

            if self.config.data.get('cutout', False):
                aug.append(Cutout(n_holes=1, length=16))

            aug.append(
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            )
            transform_train = transforms.Compose(aug)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            train_dataset = CIFAR100(root='/mnt/lustre/yuankun/transfer_datasets/cifar100/', 
                                    train=True, download=False, transform=transform_train)
            val_dataset = CIFAR100(root='/mnt/lustre/yuankun/transfer_datasets/cifar100/', 
                                    train=False, download=False, transform=transform_test)
        
        else:
            raise RuntimeError('unknown task: {}'.format(self.config.data.task))

        train_sampler = DistributedGivenIterationSampler(
                    train_dataset, 
                    self.config.data.max_iter, 
                    self.config.data.batch_size, 
                    last_iter=self.config.data.last_iter)

        val_sampler = DistributedSampler(val_dataset, round_up=False)

        train_loader = DataLoader(
                    train_dataset, 
                    batch_size=self.config.data.batch_size, 
                    shuffle=False,
                    num_workers=self.config.data.workers, 
                    pin_memory=True, 
                    sampler=train_sampler)

        val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.data.batch_size, 
                shuffle=False,
                num_workers=self.config.data.workers, 
                pin_memory=True, 
                sampler=val_sampler)

        self.train_data = {'loader': train_loader}
        self.val_data = {'loader': val_loader}


@link_dist
def main():
    parser = argparse.ArgumentParser(description='base solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--recover', type=str, default='')
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    # build or recover solver
    solver = CifarClsSolver(args.config, recover=args.recover)

    # evaluate or train
    if args.evaluate:
        if not args.recover:
            solver.logger.warn('evaluating without recovring any solver checkpoints')
        solver.evaluate()
    else:
        solver.train()


if __name__ == '__main__':
    main()