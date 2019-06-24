import os
import argparse
import pickle
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint

from solver.base_solver import BaseSolver
from config import parse_config
from utils.dist import dist_init, dist_finalize
from utils.misc import makedir, create_logger, get_logger, count_params, count_flops
from model import model_entry


class ClsSolver(BaseSolver):

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = dist_init()
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
        self.logger.info('config: {}'.format(pprint.pformat(self.config)))
        self.logger.info('hostnames: {}'.format(os.environ['SLURM_NODELIST']))

    def build_model(self):
        model = model_entry(self.config.model)
        model.cuda()

        count_params(model)
        count_flops(model, input_shape=[1, 3, self.config.data.input_size, self.config.data.input_size])

    def train(self):
        pass

    def __del__(self):
        dist_finalize()


def main():
    parser = argparse.ArgumentParser(description='base solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--recover', type=str, default='')
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    # build or recover solver
    if args.recover:
        with open(args.recover, 'rb') as f:
            solver = pickle.load(f)
    else:
        solver = ClsSolver(args.config)

    # evaluate or train
    if args.evaluate:
        if not args.recover:
            solver.logger.warn('evaluating without recovring any solver checkpoints')
        solver.evaluate()
    else:
        solver.train()


if __name__ == '__main__':
    main()
