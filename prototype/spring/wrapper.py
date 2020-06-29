import os
import argparse
from easydict import EasyDict
import time
import datetime
import torch
import copy
import yaml
import shutil
import json
import numpy as np
import linklink as link

from prototype.config import parse_config
from prototype.utils.dist import link_dist, DistModule
from prototype.utils.misc import accuracy, load_state_model, count_params, count_flops
from prototype.model import model_entry
from prototype.optimizer import FusedFP16SGD, SGD, Adam, FP16RMSprop
from prototype.solver.cls_solver import ClsSolver
from prototype.data import make_imagenet_val_data
from prototype.utils.user_analysis_helper import send_info

try:
    from SpringCommonInterface import Metric, SpringCommonInterface
except ImportError:
    SpringCommonInterface = object
    Metric = object


class ClsMetric(Metric):
    def __init__(self, top1, top5, loss):
        self.top1 = top1
        self.top5 = top5
        self.loss = loss

    def __str__(self):
        return f'top1={self.top1} top5={self.top5} loss={self.loss}'

    def __repr__(self):
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


class ClsSpringCommonInterface(ClsSolver, SpringCommonInterface):

    external_model_builder = {}

    def __init__(self, config=None, work_dir=None, metric_dict=None, ckpt_dict=None, recover=''):
        self.prototype_info = EasyDict()
        self.work_dir = work_dir
        self.metric_dict = metric_dict
        self.config_file = os.path.join(work_dir, 'config.yaml')
        self.recover = recover
        self.config = copy.deepcopy(EasyDict(config))
        self.config_copy = copy.deepcopy(self.config)
        self.setup_env()
        # set recover with ckpt_dict here
        if ckpt_dict:
            self.state = ckpt_dict
            self.logger.info(f"======= recovering from ckpt_dict, keys={list(self.state.keys())}... =======")

        self.build_model()
        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])
        self.build_optimizer()
        self.build_lr_scheduler()
        self.build_data()
        self.pre_train()
        self.curr_step = self.state['last_iter']
        if self.curr_step < self.config.data.max_iter:
            self.total_step = len(self.train_data['loader'])
            self.train_data['iter'] = None
        else:
            self.total_step = self.config.data.max_iter
            self.logger.info(
                f"======= recovering from the max_iter: {self.config.data.max_iter} =======")

        self.val_data['iter'] = None
        self.end_time = time.time()
        send_info(self.prototype_info)

    def build_model(self):
        '''overwrite build_model function in ClsSolver for interface definition of not loading weights
        '''
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.info('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        self.model = model_entry(self.config.model)
        self.model.cuda()

        count_params(self.model)
        count_flops(self.model, input_shape=[
                    1, 3, self.config.data.input_size, self.config.data.input_size])

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
                link.fp16.register_float_module(
                    link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(
                    torch.nn.BatchNorm2d, cast_args=False)
            if self.config.optimizer.get('fp16_normal_fc', False):
                self.logger.info('using normal fc for fp16')
                link.fp16.register_float_module(
                    torch.nn.Linear, cast_args=True)
            link.fp16.init()
            self.model.half()

        self.model = DistModule(self.model, self.config.dist.sync)

        return self.model

    def build_data(self):
        super().build_data()
        self.arch_data = make_imagenet_val_data(self.config.data, periodic=True)
        self.arch_data['iter'] = None
        # next update
        # self.data_loaders['train'] = self.train_data
        # self.data_loaders['val'] = self.val_data
        # self.data_loaders['arch'] = self.arch_data

    def get_batch(self, batch_type='train'):
        """data: train, val, test or arch (for NAS)

        Return:
            (batch, label): tuple.
        """
        if batch_type == 'train':
            if not self.train_data['iter']:
                self.train_data['iter'] = iter(self.train_data['loader'])
            input, target = next(self.train_data['iter'])

        elif batch_type == 'val':
            if not self.val_data['iter']:
                self.val_data['iter'] = iter(self.val_data['loader'])
            input, target = next(self.val_data['iter'])

        elif batch_type == 'arch':
            if not self.arch_data['iter']:
                self.arch_data['iter'] = iter(self.arch_data['loader'])
            input, target = next(self.arch_data['iter'])

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
        loss = self.criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))

        reduced_loss = loss.clone() / self.dist.world_size
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
        elif FP16RMSprop is not None and isinstance(self.optimizer, FP16RMSprop):
            self.optimizer.backward(loss)
            self.model.sync_gradients()
        elif isinstance(self.optimizer, SGD) or isinstance(self.optimizer, Adam):
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

        # set metric_dict
        if self.metric_dict is not None and 'eta' in self.metric_dict:
            self.metric_dict['eta'].set(self.get_eta())
        if self.metric_dict is not None and 'progress' in self.metric_dict:
            self.metric_dict['progress'].set(self.get_progress())
        self.meters.batch_time.update(time.time() - self.end_time)
        self.end_time = time.time()

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
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
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
                val_loss, prec1, prec5 = super().evaluate()
                if self.ema is not None:
                    self.ema.load_ema(self.model)
                    val_loss_ema, prec1_ema, prec5_ema = super().evaluate()
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

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.lr_scheduler

    def get_dummy_input(self):
        input = torch.zeros(1, 3, self.config.data.input_size, self.config.data.input_size)
        return input.cuda().half() if self.fp16 else input.cuda()

    def get_total_iter(self):
        return self.config.data.max_iter

    def evaluate(self):
        loss, top1, top5 = super().evaluate()
        metric = ClsMetric(top1, top5, loss)
        return metric

    @staticmethod
    def load_weights(model, ckpt_dict):
        model.load_state_dict(ckpt_dict['model'], strict=True)

    @staticmethod
    def build_model_helper(config_dict=None):
        if not isinstance(config_dict, EasyDict):
            config_dict = EasyDict(config_dict)
        model = model_entry(config_dict.model)
        model.cuda()

        # handle fp16
        if config_dict.optimizer.type == 'FP16SGD' or \
           config_dict.optimizer.type == 'FusedFP16SGD' or \
           config_dict.optimizer.type == 'FP16RMSprop':
            fp16 = True
        else:
            fp16 = False

        if fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            if config_dict.optimizer.get('fp16_normal_bn', False):
                link.fp16.register_float_module(link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(torch.nn.BatchNorm2d, cast_args=False)
            if config_dict.optimizer.get('fp16_normal_fc', False):
                link.fp16.register_float_module(torch.nn.Linear, cast_args=True)
            link.fp16.init()
            model.half()

        model = DistModule(model, config_dict.dist.sync)
        return model

    def show_log(self):
        curr_step = self.curr_step
        current_lr = self.lr_scheduler.get_lr()[0]
        remain_secs = (self.total_step - curr_step) * self.meters.batch_time.avg
        remain_time = datetime.timedelta(seconds=round(remain_secs))
        finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
        log_msg = f'Iter: [{curr_step}/{self.total_step}]\t' \
            f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
            f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
            f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
            f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
            f'Prec@5 {self.meters.top5.val:.3f} ({self.meters.top5.avg:.3f})\t' \
            f'LR {current_lr:.4f}\t' \
            f'Remaining Time {remain_time} ({finish_time})' \

        self.logger.info(log_msg)

    @classmethod
    def add_external_model(cls, name, callable_object):
        '''Add external model into the element. After this interface is called, the element should
        be able to build model by the given ``name``.

        e.g. ::

            task_helper.add_external_model('mbv2', MobileNetV2)

        Then for prototype, if the model field in yaml is `mbv2`, a MobileNetV2 instance should be built successfully.

        Args:
            name (str): The identifier of callable_object
            callable_object (callable object): A class or a function that is callable to build a torch.nn.Module model.
        '''
        cls.external_model_builder[name] = callable_object

    def to_caffe(self, save_prefix='model', input_size=None):
        from spring.nart.tools import pytorch

        with pytorch.convert_mode():
            pytorch.convert(self.model,
                            [(3, self.config.data.input_size,
                              self.config.data.input_size)],
                            filename=save_prefix,
                            input_names=['data'],
                            output_names=['out'])

    def to_kestrel(self, save_to=None):
        prefix = 'model'
        self.to_caffe(prefix)

        prototxt = '{}.prototxt'.format(prefix)
        caffemodel = '{}.caffemodel'.format(prefix)
        version = '1.0.0'
        model_name = self.config.model.type

        kestrel_model = '{}_{}.tar'.format(model_name, version)
        to_kestrel_yml = 'temp_to_kestrel.yml'
        kestrel_param = self.get_kestrel_parameter()
        with open(to_kestrel_yml, 'w') as f:
            yaml.dump(json.loads(kestrel_param), f)

        cmd = 'python -m spring.nart.tools.kestrel.classifier {} {} -v {} -c {} -n {}'.format(
            prototxt, caffemodel, version, to_kestrel_yml, model_name)

        os.system(cmd)

        if save_to is None:
            save_to = kestrel_model
        else:
            save_to = os.path.join(save_to, kestrel_model)

        shutil.move(kestrel_model, save_to)
        self.logger.info('save kestrel model to: {}'.format(save_to))

    def convert_model(self, type='caffe'):
        '''Dump pytorch model to deployable type model (skme or caffe).
        More about SKME: https://confluence.sensetime.com/pages/viewpage.action?pageId=135889068

        Args:
            type: deploy model type. "skme" or "caffe"

        Returns:
            dict: A dict of model file (string) and type.
            The format should be {"model": [model1, model2, ...], "type": "skme"}.
        '''
        assert type == 'caffe', 'only caffe model support for now'
        return {
            'model': [self.to_caffe()],
            'type': 'caffe'
        }

    def get_kestrel_parameter(self):
        '''For Classifier Plugin'''
        kestrel_param = EasyDict()
        kestrel_param['pixel_means'] = self.config.get('pixel_means', [124.16, 116.736, 103.936])
        kestrel_param['pixel_stds'] = self.config.get('pixel_stds', [58.624, 57.344, 57.6])
        kestrel_param['is_rgb'] = self.config.get('is_rgb', True)
        kestrel_param['save_all_label'] = self.config.get('save_all_label', True)
        kestrel_param['type'] = self.config.get('type', 'UNKNOWN')
        if hasattr(self.config, 'class_label'):
            kestrel_param['class_label'] = self.config['class_label']
        else:
            kestrel_param['class_label'] = {}
            kestrel_param['class_label']['imagenet'] = {}
            kestrel_param['class_label']['imagenet']['calculator'] = 'bypass'
            num_classes = self.config.model.get('num_classes', 1000)
            kestrel_param['class_label']['imagenet']['labels'] = [
                str(i) for i in np.arange(num_classes)]
            kestrel_param['class_label']['imagenet']['feature_start'] = 0
            kestrel_param['class_label']['imagenet']['feature_end'] = num_classes - 1

        return json.dumps(kestrel_param)


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
    solver = ClsSpringCommonInterface(config=config, work_dir=work_dir, recover=args.recover)

    # evaluate or train
    if args.evaluate:
        if not args.recover:
            solver.logger.warn('evaluating without recovring any solver checkpoints')
        solver.evaluate()
    else:
        if solver.config.data.last_iter < solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
