import os
import logging
import torch
import linklink as link

from utils.dist import simple_group_split


def makedir(path):
    if link.get_rank() == 0 and not os.path.exists(path):
        os.makedirs(path)
    link.barrier()


class RankFilter(logging.Filter):
    def filter(self, record):
        return False


def create_logger(log_file, level=logging.INFO):
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


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
