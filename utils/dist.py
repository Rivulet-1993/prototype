import os
import torch
import numpy as np
import linklink as link


def dist_init(method='slurm', device_id=0):
    if method == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        # ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
    elif method == 'single_node':
        torch.cuda.set_device(device_id)

    link.initialize()
    world_size = link.get_world_size()
    rank = link.get_rank()

    return rank, world_size


def dist_finalize():
    link.finalize()


def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank//group_size]
