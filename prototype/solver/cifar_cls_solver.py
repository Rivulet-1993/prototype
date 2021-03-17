import torch
import linklink as link
import time
import datetime
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

from .cls_solver import ClsSolver
from prototype.utils.dist import link_dist
from prototype.data.sampler import DistributedGivenIterationSampler, DistributedSampler
from prototype.data.auto_augmentation import CIFAR10Policy, Cutout
from prototype.utils.misc import accuracy, AverageMeter, mixup_data, cutmix_data, mix_criterion
from prototype.optimizer import FP16RMSprop, FP16SGD, FusedFP16SGD


class CifarClsSolver(ClsSolver):

    def build_data(self):
        """
        Specific for CIFAR10/CIFAR100
        """
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

    def train(self):

        self.pre_train()
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()
        for i, (input, target) in enumerate(self.train_data['loader']):
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            target = target.squeeze().cuda().long()
            input = input.cuda().half() if self.fp16 else input.cuda()
            # mixup
            if self.mixup < 1.0:
                input, target_a, target_b, lam = mixup_data(input, target, self.mixup)
            # cutmix
            if self.cutmix > 0.0:
                input, target_a, target_b, lam = cutmix_data(input, target, self.cutmix)
            # forward
            logits = self.model(input)
            # mixup
            if self.mixup < 1.0 or self.cutmix > 0.0:
                loss = mix_criterion(self.criterion, logits, target_a, target_b, lam)
                loss /= self.dist.world_size
            else:
                loss = self.criterion(logits, target) / self.dist.world_size

            # measure accuracy and record loss
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            reduced_loss = loss.clone()
            reduced_prec1 = prec1.clone() / self.dist.world_size
            reduced_prec5 = prec5.clone() / self.dist.world_size

            self.meters.losses.reduce_update(reduced_loss)
            self.meters.top1.reduce_update(reduced_prec1)
            self.meters.top5.reduce_update(reduced_prec5)
            self.optimizer.zero_grad()
            if FusedFP16SGD is not None and isinstance(self.optimizer, FusedFP16SGD):
                self.optimizer.backward(loss)
                self.model.sync_gradients()
                self.optimizer.step()
            elif isinstance(self.optimizer, FP16SGD) or isinstance(self.optimizer, FP16RMSprop):

                def closure():
                    self.optimizer.backward(loss, False)
                    self.model.sync_gradients()
                    # check overflow, convert to fp32 grads, downscale
                    self.optimizer.update_master_grads()
                    return loss

                self.optimizer.step(closure)
            else:
                loss.backward()
                self.model.sync_gradients()
                self.optimizer.step()

            # EMA
            if self.ema is not None:
                self.ema.step(self.model, curr_step=curr_step)

            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                    f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                    f'LR {current_lr:.4f}\t' \
                    f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                val_loss, prec1, prec5 = self.evaluate()
                if self.ema is not None:
                    self.ema.load_ema(self.model)
                    val_loss_ema, prec1_ema, prec5_ema = self.evaluate()
                    self.ema.recover(self.model)
                    if self.dist.rank == 0:
                        self.tb_logger.add_scalars('loss_val', {'ema': val_loss_ema}, curr_step)
                        self.tb_logger.add_scalars('acc1_val', {'ema': prec1_ema}, curr_step)
                        self.tb_logger.add_scalars('acc5_val', {'ema': prec5_ema}, curr_step)

                if self.dist.rank == 0:
                    self.tb_logger.add_scalar('loss_val', val_loss, curr_step)
                    self.tb_logger.add_scalar('acc1_val', prec1, curr_step)
                    self.tb_logger.add_scalar('acc5_val', prec5, curr_step)

                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'

                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step
                    if self.ema is not None:
                        self.state['ema'] = self.ema.state_dict()

                    torch.save(self.state, ckpt_name)

            end = time.time()

    @torch.no_grad()
    def evaluate(self):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)
        top5 = AverageMeter(0)

        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        val_iter = len(self.val_data['loader'])
        end = time.time()
        for i, (input, target) in enumerate(self.val_data['loader']):
            input = input.cuda().half() if self.fp16 else input.cuda()
            target = target.squeeze().view(-1).cuda().long()
            logits = self.model(input)
            # measure accuracy and record loss
            # / world_size # loss should not be scaled here, it's reduced later!
            loss = criterion(logits, target)
            prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
            num = input.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top5.update(prec5.item(), num)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if (i+1) % self.config.saver.print_freq == 0:
                self.logger.info(f'Test: [{i+1}/{val_iter}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})')

        # gather final results
        total_num = torch.Tensor([losses.count])
        loss_sum = torch.Tensor([losses.avg*losses.count])
        top1_sum = torch.Tensor([top1.avg*top1.count])
        top5_sum = torch.Tensor([top5.avg*top5.count])
        link.allreduce(total_num)
        link.allreduce(loss_sum)
        link.allreduce(top1_sum)
        link.allreduce(top5_sum)
        final_loss = loss_sum.item()/total_num.item()
        final_top1 = top1_sum.item()/total_num.item()
        final_top5 = top5_sum.item()/total_num.item()
        self.logger.info(f' * Prec@1 {final_top1:.3f}\tPrec@5 {final_top5:.3f}\t\
            Loss {final_loss:.3f}\ttotal_num={total_num.item()}')

        self.model.train()

        return final_loss, final_top1, final_top5


@link_dist
def main():
    parser = argparse.ArgumentParser(description='cifar solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    # build or recover solver
    solver = CifarClsSolver(args.config)

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
