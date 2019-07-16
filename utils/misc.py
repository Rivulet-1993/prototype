import os
import logging
import torch
import linklink as link
from collections import defaultdict
import numpy as np

from utils.dist import simple_group_split

_logger = None
_log_file = None


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        link.allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count


def makedir(path):
    if link.get_rank() == 0 and not os.path.exists(path):
        os.makedirs(path)
    link.barrier()


class RankFilter(logging.Filter):
    def filter(self, record):
        return False


def create_logger(log_file, level=logging.INFO):
    global _logger, _log_file
    if _logger is None:
        _logger = logging.getLogger()
        formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        _logger.setLevel(level)
        _logger.addHandler(fh)
        _logger.addHandler(sh)
        _log_file = log_file
    elif _log_file != log_file:
        raise RuntimeError(f'calling create_logger with log_file={log_file}, '
                           f'but previously called with log_file={_log_file}')
    return _logger


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if link.get_rank() > 0:
        logger.addFilter(RankFilter())
    return logger


def get_bn(config):
    if config.use_sync_bn:
        group_size = config.kwargs.group_size
        var_mode = config.kwargs.var_mode
        if group_size == 1:
            bn_group = None
        else:
            world_size, rank = link.get_world_size(), link.get_rank()
            assert world_size % group_size == 0
            bn_group = simple_group_split(world_size, rank, world_size // group_size)

        del config.kwargs['group_size']
        config.kwargs.group = bn_group
        config.kwargs.var_mode = (link.syncbnVarMode_t.L1 if var_mode == 'L1' else link.syncbnVarMode_t.L2)

        def BNFunc(*args, **kwargs):
            return link.nn.SyncBatchNorm2d(*args, **kwargs, **config.kwargs)

        return BNFunc
    else:
        def BNFunc(*args, **kwargs):
            return torch.nn.BatchNorm2d(*args, **kwargs, **config.kwargs)
        return BNFunc


def count_params(model):
    logger = get_logger(__name__)

    total = sum(p.numel() for p in model.parameters())
    conv = 0
    fc = 0
    others = 0
    for name, m in model.named_modules():
        # skip non-leaf modules
        if len(list(m.children())) > 0:
            continue
        num = sum(p.numel() for p in m.parameters())
        if isinstance(m, torch.nn.Conv2d):
            conv += num
        elif isinstance(m, torch.nn.Linear):
            fc += num
        else:
            others += num

    M = 1e6

    logger.info('total param: {:.3f}M, conv: {:.3f}M, fc: {:.3f}M, others: {:.3f}M'
                .format(total/M, conv/M, fc/M, others/M))


def count_flops(model, input_shape):
    logger = get_logger(__name__)

    flops_dict = {}
    def make_conv2d_hook(name):

        def conv2d_hook(m, input):
            n, _, h, w = input[0].size(0), input[0].size(1), input[0].size(2), input[0].size(3)
            flops = n * h * w * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] \
                / m.stride[1] / m.stride[1] / m.groups
            flops_dict[name] = int(flops)

        return conv2d_hook

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            h = m.register_forward_pre_hook(make_conv2d_hook(name))
            hooks.append(h)

    input = torch.zeros(*input_shape).cuda()
    with torch.no_grad():
        _ = model(input)

    total_flops = 0
    for k, v in flops_dict.items():
        # logger.info('module {}: {}'.format(k, v))
        total_flops += v
    logger.info('total FLOPS: {:.2f}M'.format(total_flops/1e6))

    for h in hooks:
        h.remove()


def param_group_all(model, config):
    logger = get_logger(__name__)
    pgroup_normal = []
    pgroup = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    names = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    if 'conv_dw_w' in config:
        pgroup['conv_dw_w'] = []
        names['conv_dw_w'] = []
    if 'conv_dw_b' in config:
        pgroup['conv_dw_b'] = []
        names['conv_dw_b'] = []
    if 'conv_dense_w' in config:
        pgroup['conv_dense_w'] = []
        names['conv_dense_w'] = []
    if 'conv_dense_b' in config:
        pgroup['conv_dense_b'] = []
        names['conv_dense_b'] = []
    if 'linear_w' in config:
        pgroup['linear_w'] = []
        names['linear_w'] = []

    names_all = []
    type2num = defaultdict(lambda: 0)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                if 'conv_dw_b' in pgroup and m.groups == m.in_channels:
                    pgroup['conv_dw_b'].append(m.bias)
                    names_all.append(name+'.bias')
                    names['conv_dw_b'].append(name+'.bias')
                    type2num[m.__class__.__name__+'.bias(dw)'] += 1
                elif 'conv_dense_b' in pgroup and m.groups == 1:
                    pgroup['conv_dense_b'].append(m.bias)
                    names_all.append(name+'.bias')
                    names['conv_dense_b'].append(name+'.bias')
                    type2num[m.__class__.__name__+'.bias(dense)'] += 1
                else:
                    pgroup['conv_b'].append(m.bias)
                    names_all.append(name+'.bias')
                    names['conv_b'].append(name+'.bias')
                    type2num[m.__class__.__name__+'.bias'] += 1
            if 'conv_dw_w' in pgroup and m.groups == m.in_channels:
                pgroup['conv_dw_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['conv_dw_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight(dw)'] += 1
            elif 'conv_dense_w' in pgroup and m.groups == 1:
                pgroup['conv_dense_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['conv_dense_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight(dense)'] += 1

        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                pgroup['linear_b'].append(m.bias)
                names_all.append(name+'.bias')
                names['linear_b'].append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1
            if 'linear_w' in pgroup:
                pgroup['linear_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['linear_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight'] += 1
        elif (isinstance(m, torch.nn.BatchNorm2d)
              or isinstance(m, torch.nn.BatchNorm1d)
              or isinstance(m, link.nn.SyncBatchNorm2d)):
            if m.weight is not None:
                pgroup['bn_w'].append(m.weight)
                names_all.append(name+'.weight')
                names['bn_w'].append(name+'.weight')
                type2num[m.__class__.__name__+'.weight'] += 1
            if m.bias is not None:
                pgroup['bn_b'].append(m.bias)
                names_all.append(name+'.bias')
                names['bn_b'].append(name+'.bias')
                type2num[m.__class__.__name__+'.bias'] += 1

    for name, p in model.named_parameters():
        if name not in names_all:
            pgroup_normal.append(p)

    param_groups = [{'params': pgroup_normal}]
    for ptype in pgroup.keys():
        if ptype in config.keys():
            param_groups.append({'params': pgroup[ptype], **config[ptype]})
        else:
            param_groups.append({'params': pgroup[ptype]})

        logger.info(ptype)
        for k, v in param_groups[-1].items():
            if k == 'params':
                logger.info('   params: {}'.format(len(v)))
            else:
                logger.info('   {}: {}'.format(k, v))

    for ptype, pconf in config.items():
        logger.info('names for {}({}): {}'.format(ptype, len(names[ptype]), names[ptype]))

    return param_groups, type2num


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_state_model(model, state):

    logger = get_logger(__name__)
    logger.info('======= loading model state... =======')

    model.load_state_dict(state, strict=False)

    state_keys = set(state.keys())
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - state_keys
    for k in missing_keys:
        logger.warn(f'missing key: {k}')


def load_state_optimizer(optimizer, state):

    logger = get_logger(__name__)
    logger.info('======= loading optimizer state... =======')

    optimizer.load_state_dict(state)
