import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

from .cls_solver import ClsSolver
from prototype.utils.dist import link_dist
from prototype.data.sampler import DistributedGivenIterationSampler, DistributedSampler
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
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            )
            transform_train = transforms.Compose(aug)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_dataset = CIFAR10(root='/mnt/lustre/share/prototype_cifar/cifar10/',
                                    train=True, download=False, transform=transform_train)
            val_dataset = CIFAR10(root='/mnt/lustre/share/prototype_cifar/cifar10/',
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
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            )
            transform_train = transforms.Compose(aug)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])

            train_dataset = CIFAR100(root='/mnt/lustre/share/prototype_cifar/cifar100/',
                                     train=True, download=False, transform=transform_train)
            val_dataset = CIFAR100(root='/mnt/lustre/share/prototype_cifar/cifar100/',
                                   train=False, download=False, transform=transform_test)

        else:
            raise RuntimeError(
                'unknown task: {}'.format(self.config.data.task))

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
        if solver.ema is not None:
            solver.ema.load_ema(solver.model)
            solver.evaluate()
    else:
        if solver.config.data.last_iter < solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
