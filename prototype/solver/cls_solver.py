import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time
import datetime
import torch
import linklink as link

from .base_solver import BaseSolver
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


class ClsSolver(BaseSolver):

    def __init__(self, config_file, recover=''):
        self.config_file = config_file
        self.recover = recover
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()
        self.build_optimizer()
        self.build_lr_scheduler()
        self.build_data()

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        # tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.info(f'config: {pprint.pformat(self.config)}')
        self.logger.info(f"hostnames: {os.environ['SLURM_NODELIST']}")
        # recover
        if self.recover:
            self.state = torch.load(self.recover, 'cpu')
            self.logger.info(f"======= recovering from {self.recover}, keys={list(self.state.keys())}... =======")
        else:
            self.state = {}
            self.state['last_iter'] = 0
        # others
        torch.backends.cudnn.benchmark = True

    def build_model(self):
        self.model = model_entry(self.config.model)
        self.model.cuda()

        count_params(self.model)
        count_flops(self.model, input_shape=[1, 3, self.config.data.input_size, self.config.data.input_size])

        # handle fp16
        if self.config.optimizer.type == 'FP16SGD' or \
           self.config.optimizer.type == 'FusedFP16SGD' or \
           self.config.optimizer.type == 'FP16RMSprop':
            self.fp16 = True
        else:
            self.fp16 = False

        if self.fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            if self.config.optimizer.get('fp16_normal_bn', False):
                self.logger.info('using normal bn for fp16')
                link.fp16.register_float_module(link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(torch.nn.BatchNorm2d, cast_args=False)
            if self.config.optimizer.get('fp16_normal_fc', False):
                self.logger.info('using normal fc for fp16')
                link.fp16.register_float_module(torch.nn.Linear, cast_args=True)
            link.fp16.init()
            self.model.half()

        self.model = DistModule(self.model, self.config.dist.sync)

        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    def build_optimizer(self):

        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr

        # make param_groups
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(self.model, pconfig)

        opt_config.kwargs.params = param_group

        self.optimizer = optim_entry(opt_config)

        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

        # EMA
        if self.config.ema.enable:
            self.config.ema.kwargs.model = self.model
            self.ema = EMA(**self.config.ema.kwargs)
        else:
            self.ema = None

        if 'ema' in self.state:
            self.ema.load_state_dict(self.state['ema'])

    def build_lr_scheduler(self):
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer.optimizer if isinstance(self.optimizer, FP16SGD) or \
                                                        isinstance(self.optimizer, FP16RMSprop) else self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
        self.config.data.last_iter = self.state['last_iter']

        if self.config.data.max_iter != self.config.data.last_iter:
            self.train_data = make_imagenet_train_data(self.config.data)
        else:
            self.logger.info(
                f"======= recovering from the max_iter: {self.config.data.max_iter} =======")

        self.val_data = make_imagenet_val_data(self.config.data)

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.print_freq)
        self.meters.step_time = AverageMeter(self.config.print_freq)
        self.meters.data_time = AverageMeter(self.config.print_freq)
        self.meters.losses = AverageMeter(self.config.print_freq)
        self.meters.top1 = AverageMeter(self.config.print_freq)
        self.meters.top5 = AverageMeter(self.config.print_freq)

        self.model.train()

        label_smooth = self.config.get('label_smooth', 0.0)
        num_classes = self.config.get('num_classes', 1000)
        if label_smooth > 0:
            self.logger.info('using label_smooth: {}'.format(label_smooth))
            self.criterion = LabelSmoothCELoss(label_smooth, num_classes)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

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

            # forward
            logits = self.model(input)
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

            if curr_step % self.config.print_freq == 0 and self.dist.rank == 0:
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

            if curr_step > 0 and curr_step % self.config.val_freq == 0:
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
                    if self.config.save_many:
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

    def evaluate(self, fusion_list=None, fuse_prob=False):
        batch_time = AverageMeter(0)
        losses = AverageMeter(0)
        top1 = AverageMeter(0)
        top5 = AverageMeter(0)

        # switch to evaluate mode
        # if fusion_list is not None:
        #     model_list = []
        #     for i in range(len(fusion_list)):
        #         model_list.append(model_entry(self.config.model))
        #         model_list[i].cuda()
        #         model_list[i] = DistModule(model_list[i], args.sync)
        #         load_state(fusion_list[i], model_list[i])
        #         model_list[i].eval()
        #     if fuse_prob:
        #         softmax = nn.Softmax(dim=1)
        # else:
        self.model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        val_iter = len(self.val_data['loader'])

        end = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(self.val_data['loader']):
                input = input.cuda().half() if self.fp16 else input.cuda()
                target = target.squeeze().view(-1).cuda().long()
                # compute output
                # if fusion_list is not None:
                #     output_list = []
                #     for model_idx in range(len(fusion_list)):
                #         tmp = model_list[model_idx](input)
                #         if isinstance(tmp, dict):
                #             output = tmp['logits']
                #         else:
                #             output = tmp
                #         if fuse_prob:
                #             output = softmax(output)
                #         output_list.append(output)
                #     output = torch.stack(output_list, 0)
                #     output = torch.mean(output, 0)
                # else:
                logits = self.model(input)

                # measure accuracy and record loss
                loss = criterion(logits, target)  # / world_size # loss should not be scaled here, it's reduced later!
                prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))

                num = input.size(0)
                losses.update(loss.item(), num)
                top1.update(prec1.item(), num)
                top5.update(prec5.item(), num)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if (i+1) % self.config.print_freq == 0:
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
    parser = argparse.ArgumentParser(description='base solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--recover', type=str, default='')
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    # build or recover solver
    solver = ClsSolver(args.config, recover=args.recover)

    # evaluate or train
    if args.evaluate:
        if not args.recover:
            solver.logger.warn('evaluating without recovring any solver checkpoints')
        solver.evaluate()
    else:
        solver.train()


if __name__ == '__main__':
    main()
