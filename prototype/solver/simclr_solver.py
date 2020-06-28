import argparse
import time
import datetime

import torch
import linklink as link
import torch.nn.functional as F

from prototype.model import model_entry
from prototype.solver.cls_solver import ClsSolver
from prototype.utils.simclr_builder import SimCLR
from prototype.utils.dist import link_dist, DistModule
from prototype.utils.misc import count_params, count_flops, load_state_model
from prototype.optimizer import FP16RMSprop, FP16SGD, FusedFP16SGD
from prototype.loss_functions import NT_Xent


class SimCLRSolver(ClsSolver):

    def build_model(self):
        encoder = model_entry(self.config.model)
        self.model = SimCLR(encoder)
        self.model.cuda()
        count_params(self.model.encoder)
        count_flops(self.model.encoder, input_shape=[
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

        if 'model' in self.state:
            load_state_model(self.model, self.state['model'])

    def pre_train(self):
        super().pre_train()
        self.criterion = NT_Xent(self.config.data.batch_size, self.config.temperature)

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
            input = input.cuda().half() if self.fp16 else input.cuda()

            # forward
            z_i, z_j = self.model(input)
            # normalize projection feature vectors
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

            loss = self.criterion(z_i, z_j) / self.dist.world_size

            reduced_loss = loss.clone()
            self.meters.losses.reduce_update(reduced_loss)

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

            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)

            if curr_step % self.config.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar(
                    'loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (total_step - curr_step) * \
                    self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'LR {current_lr:.4f}\t' \
                    f'Remaining Time {remain_time} ({finish_time})'

                self.logger.info(log_msg)

            if curr_step > 0 and curr_step % self.config.save_freq == 0:
                if self.dist.rank == 0:
                    if self.config.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'

                    self.state['model'] = self.model.state_dict()
                    self.state['optimizer'] = self.optimizer.state_dict()
                    self.state['last_iter'] = curr_step

                    torch.save(self.state, ckpt_name)

            end = time.time()


@link_dist
def main():
    parser = argparse.ArgumentParser(description='simclr solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--recover', type=str, default='')

    args = parser.parse_args()

    # build or recover solver
    solver = SimCLRSolver(args.config, recover=args.recover)

    if solver.config.data.last_iter < solver.config.data.max_iter:
        solver.train()
    else:
        solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
