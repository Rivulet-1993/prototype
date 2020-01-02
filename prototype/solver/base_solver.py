import argparse
import pickle
from pprint import pprint
from prototype.config import parse_config


class BaseSolver(object):

    def __init__(self, config_file):
        config = parse_config(config_file)
        pprint(config)

    def setup_envs(self):
        # dist
        # directories
        # logger
        # tb_logger
        pass

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    def train(self):
        pass


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
        solver = BaseSolver(args.config)

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
