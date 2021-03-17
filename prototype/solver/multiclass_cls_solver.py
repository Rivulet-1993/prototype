import argparse
import time
import datetime
import os
import json

import torch
import torch.nn.functional as F
import linklink as link

from .cls_solver import ClsSolver
from prototype.utils.dist import link_dist, broadcast_object
from prototype.utils.misc import accuracy
from prototype.optimizer import optim_entry, FP16RMSprop, FP16SGD, FusedFP16SGD # noqa


class MultiClsSolver(ClsSolver):

    def build_model(self):
        super().build_model()
        self.feature_dims = self.config.model.feature_dims
        assert sum(self.feature_dims) == self.config.model.kwargs.num_classes

    def train(self):
        self.pre_train()
        total_step = len(self.train_data['loader'])
        start_step = self.state['last_iter'] + 1
        end = time.time()
        for i, batch in enumerate(self.train_data['loader']):
            input = batch['image']
            label_list = batch['label_list']
            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # transfer input to gpu
            label_list = [_label.squeeze().cuda().long() for _label in label_list]
            input = input.cuda().half() if self.fp16 else input.cuda()
            # forward
            logits = self.model(input)
            multi_logits = logits.split(self.feature_dims, dim=1)
            # compute aggregation loss and average accuracy for multiple classes

            for j, (s_logits, s_label) in enumerate(zip(multi_logits, label_list)):
                if j == 0:
                    loss = self.criterion(s_logits, s_label) / self.dist.world_size
                    prec1 = accuracy(s_logits, s_label, topk=(1,))[0]
                else:
                    loss += self.criterion(s_logits, s_label) / self.dist.world_size
                    prec1 += accuracy(s_logits, s_label, topk=(1,))[0]
            loss /= len(self.feature_dims)
            prec1 /= len(self.feature_dims)

            reduced_loss = loss.clone()
            reduced_prec1 = prec1.clone() / self.dist.world_size
            self.meters.losses.reduce_update(reduced_loss)
            self.meters.top1.reduce_update(reduced_prec1)

            # compute and update gradient
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

            # training logger
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar('loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar('acc1_train', self.meters.top1.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                remain_secs = (total_step - curr_step) * self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'Prec@1 {self.meters.top1.val:.3f} ({self.meters.top1.avg:.3f})\t' \
                    f'LR {current_lr:.4f}\t' \
                    f'Remaining Time {remain_time} ({finish_time})'
                self.logger.info(log_msg)

            # testing during training
            if curr_step > 0 and curr_step % self.config.saver.val_freq == 0:
                metrics = self.evaluate()
                self.logger.info(json.dumps(metrics.metric, indent=2))
                if self.ema is not None:
                    self.ema.load_ema(self.model)
                    ema_metrics = self.evaluate()
                    self.logger.info(json.dumps(ema_metrics.metric, indent=2))
                    self.ema.recover(self.model)

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
        self.model.eval()
        res_file = os.path.join(self.path.result_path, f'results.txt.rank{self.dist.rank}')
        writer = open(res_file, 'w')
        for batch_idx, batch in enumerate(self.val_data['loader']):
            input = batch['image']
            input = input.cuda().half() if self.fp16 else input.cuda()
            # compute output
            logits = self.model(input)
            multi_logits = logits.split(self.feature_dims, dim=1)
            multi_scores = [F.softmax(s_logits, dim=1) for s_logits in multi_logits]
            # compute predictions for each attribute
            multi_preds = [s_logits.data.topk(k=1, dim=1)[1] for s_logits in multi_logits]
            batch.update({'prediction': multi_preds})
            batch.update({'score': multi_scores})
            # save prediction information
            self.val_data['loader'].dataset.dump(writer, batch)

        writer.close()
        link.barrier()
        if self.dist.rank == 0:
            metrics = self.val_data['loader'].dataset.evaluate(res_file)
        else:
            metrics = {}
        link.barrier()
        # broadcast metrics to other process
        metrics = broadcast_object(metrics)
        self.model.train()
        return metrics


@link_dist
def main():
    parser = argparse.ArgumentParser(description='Multiple Class Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    solver = MultiClsSolver(args.config)
    # evaluate or train
    if args.evaluate:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn('Evaluating without resuming any solver checkpoints.')
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
