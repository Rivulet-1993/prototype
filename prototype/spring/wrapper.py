import os
import argparse
from easydict import EasyDict
import time
import datetime
import torch
import copy
import linklink as link

from prototype.config import parse_config
from prototype.utils.dist import link_dist, DistModule
from prototype.utils.misc import accuracy
from prototype.model import model_entry
from prototype.optimizer import FusedFP16SGD, SGD
from prototype.solver.cls_solver import ClsSolver

from .SpringCommonInterface import Metric, SpringCommonInterface


class ClsMetric(Metric):
    def __init__(self, top1, top5, loss):
        self.top1 = top1
        self.top5 = top5
        self.loss = loss

    def __str__(self):
        return f'top1={self.top1} top5={self.top5} loss={self.loss}'

    def __eq__(self, other):
        return self.top1 == other.top1

    def __ne__(self, other):
        return self.top1 != other.top1

    def __gt__(self, other):
        return self.top1 > other.top1

    def __lt__(self, other):
        return self.top1 < other.top1

    def __ge__(self, other):
        return self.top1 >= other.top1

    def __le__(self, other):
        return self.top1 <= other.top1


class ClsSolverExchange(ClsSolver):

    def __init__(self, config=None, work_dir=None, recover_dict=None):
        self.work_dir = work_dir
        self.config_file = os.path.join(work_dir, 'config.yaml')
        self.recover = ''  # set recover to ''
        self.config = copy.deepcopy(EasyDict(config))
        self.config_copy = copy.deepcopy(self.config)
        self.setup_env()
        # set recover with recover_dict here
        if recover_dict:
            self.state = recover_dict
            self.logger.info(f"======= recovering from recover_dict, keys={list(self.state.keys())}... =======")

        self.build_model()
        self.build_optimizer()
        self.build_lr_scheduler()
        self.build_data()
        self.pre_train()
        self.curr_step = self.state['last_iter']
        self.total_step = len(self.train_data['loader'])
        self.train_data['iter'] = None
        self.val_data['iter'] = None
        self.end_time = time.time()

    def get_batch(self, batch_type='train'):
        if batch_type == 'train':
            if not self.train_data['iter']:
                self.train_data['iter'] = iter(self.train_data['loader'])
            input, target = next(self.train_data['iter'])
            target = target.squeeze().cuda().long()
            input = input.cuda().half() if self.fp16 else input.cuda()
            return input, target

    def forward(self, batch):
        self.curr_step += 1

        # measure data loading time
        self.meters.data_time.update(time.time() - self.end_time)

        input, target = batch[0], batch[1]
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

        return loss

    def backward(self, loss):
        self.optimizer.zero_grad()

        if FusedFP16SGD is not None and isinstance(self.optimizer, FusedFP16SGD):
            self.optimizer.backward(loss)
            self.model.sync_gradients()
        elif isinstance(self.optimizer, SGD):
            loss.backward()
            self.model.sync_gradients()
        else:
            raise RuntimeError(f'unknown optimizer: {self.optimizer}')

    def update(self):
        self.lr_scheduler.step(self.curr_step)
        self.optimizer.step()
        # EMA
        if self.ema is not None:
            self.ema.step(self.model, curr_step=self.curr_step)

    def get_dump_dict(self):
        state = {}
        state['config'] = self.config_copy
        state['model'] = self.model.state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['last_iter'] = self.curr_step
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        return state

    def get_eta(self):
        return (self.total_step - self.curr_step) * self.meters.batch_time.avg

    def get_progress(self):
        return self.curr_step / self.total_step * 100

    def train(self):

        for i in range(self.total_step):

            batch = self.get_batch()
            loss = self.forward(batch)
            self.backward(loss)
            self.update()

            # measure elapsed time
            self.meters.batch_time.update(time.time() - self.end_time)

            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            curr_step = self.curr_step

            if curr_step % self.config.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('acc5_train', self.meters.top5.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (self.total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                log_msg = f'Iter: [{curr_step}/{self.total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                    f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
                    f'LR {current_lr:.4f}\t' \
                    f'Remaining Time {remain_time} ({finish_time})' \

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

                    state = self.get_dump_dict()
                    torch.save(state, ckpt_name)

            self.end_time = time.time()


class ClsSpringCommonInterface(SpringCommonInterface):

    def __init__(self, config=None, work_dir=None, metric_dict=None, ckpt_dict=None):
        self.solver = ClsSolverExchange(config=config, work_dir=work_dir, recover_dict=ckpt_dict)
        self.metric_dict = metric_dict

    def get_model(self):
        return self.solver.model

    def get_optimizer(self):
        return self.solver.optimizer

    def get_scheduler(self):
        return self.solver.lr_scheduler

    def get_dummy_input(self):
        input = torch.zeros(1, 3, self.solver.config.data.input_size, self.solver.config.data.input_size)
        return input.cuda().half() if self.solver.fp16 else input.cuda()

    def get_dump_dict(self):
        return self.solver.get_dump_dict()

    def get_batch(self, batch_type='train'):
        return self.solver.get_batch(batch_type=batch_type)

    def get_total_iter(self):
        return self.solver.config.data.max_iter

    def forward(self, batch):
        return self.solver.forward(batch)

    def backward(self, loss):
        self.solver.backward(loss)

    def update(self):
        self.solver.update()
        # set metric_dict
        if self.metric_dict is not None and 'eta' in self.metric_dict:
            self.metric_dict['eta'].set(self.solver.get_eta())
        if self.metric_dict is not None and 'progress' in self.metric_dict:
            self.metric_dict['progress'].set(self.solver.get_progress())

    def train(self):
        self.solver.train()

    def evaluate(self):
        top1, top5, loss = self.solver.evaluate()
        metric = ClsMetric(top1, top5, loss)
        return metric

    @staticmethod
    def load_weights(model, ckpt_dict):
        model.load_state_dict(ckpt_dict['model'], strict=True)

    @staticmethod
    def build_model_helper(ckpt_path=None, config=None):

        assert ckpt_path != config

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, 'cpu')
            config = EasyDict(ckpt['config'])

        model = model_entry(config.model)
        model.cuda()

        # handle fp16
        if config.optimizer.type == 'FP16SGD' or \
           config.optimizer.type == 'FusedFP16SGD' or \
           config.optimizer.type == 'FP16RMSprop':
            fp16 = True
        else:
            fp16 = False

        if fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            if config.optimizer.get('fp16_normal_bn', False):
                link.fp16.register_float_module(link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(torch.nn.BatchNorm2d, cast_args=False)
            if config.optimizer.get('fp16_normal_fc', False):
                link.fp16.register_float_module(torch.nn.Linear, cast_args=True)
            link.fp16.init()
            model.half()

        model = DistModule(model, config.dist.sync)
        return model

    def build_model(self):
        self.solver.build_model()

    @property
    def logger(self):
        return self.solver.logger


@link_dist
def main():
    parser = argparse.ArgumentParser(description='base solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--recover', type=str, default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--test-sci', action='store_true')

    args = parser.parse_args()

    config = parse_config(args.config)
    work_dir = os.path.dirname(args.config)

    if args.test_sci:
        sci = ClsSpringCommonInterface(config=config, work_dir=work_dir, metric_dict=None)
        sci.logger.warn('init done')
        ckpt_dict = torch.load('checkpoints/ckpt_5000.pth.tar', 'cpu')
        sci = ClsSpringCommonInterface(config=config, work_dir=work_dir, metric_dict=None, ckpt_dict=ckpt_dict)
        sci.logger.warn('init with ckpt_dict done')
        model = sci.get_model()
        sci.logger.warn('get_model done')
        sci.get_optimizer()
        sci.logger.warn('get_optimizer done')
        sci.get_scheduler()
        sci.logger.warn('get_scheduler done')
        sci.get_dummy_input()
        sci.logger.warn('get_dummy_input done')
        sci.get_dump_dict()
        sci.logger.warn('get_dump_dict done')
        batch = sci.get_batch()
        sci.logger.warn('get_batch done')
        sci.get_total_iter()
        sci.logger.warn('get_total_iter done')
        loss = sci.forward(batch)
        sci.logger.warn('forward done')
        sci.backward(loss)
        sci.logger.warn('backward done')
        sci.update()
        sci.logger.warn('update done')
        # sci.train()
        # sci.logger.warn('train done')
        sci.evaluate()
        sci.logger.warn('evaluate done')
        sci.load_weights(model, ckpt_dict)
        sci.logger.warn('load_weights done')
        sci.build_model_helper(ckpt_path='checkpoints/ckpt_100.pth.tar')
        sci.logger.warn('build_model_helper done')
        sci.build_model()
        sci.logger.warn('build_model done')
        return

    # build or recover solver
    solver = ClsSolverExchange(config=config, work_dir=work_dir, recover=args.recover)

    # evaluate or train
    if args.evaluate:
        if not args.recover:
            solver.logger.warn('evaluating without recovring any solver checkpoints')
        solver.evaluate()
    else:
        solver.train()


if __name__ == '__main__':
    main()
