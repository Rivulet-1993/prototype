import argparse
import torch
from prototype.solver.cls_solver import ClsSolver
from prototype.utils.dist import link_dist


class LinearClsSolver(ClsSolver):

    def setup_env(self):
        super().setup_env()
        # moco pretrain model
        assert hasattr(self.config.saver, 'pretrain'), 'No pretrain model is provided!'
        model_state = torch.load(self.config.saver.pretrain.path, 'cpu')['model']
        # refactor pretrain model
        encoder_state = {}
        for key, value in model_state.items():
            if 'encoder_q' in key and 'fc' not in key:
                new_key = key.replace('encoder_q.', '')
                encoder_state[new_key] = value
        self.state = {'model': encoder_state, 'last_iter': 0}


@link_dist
def main():
    parser = argparse.ArgumentParser(description='linear protocol cls solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    # build or recover solver
    solver = LinearClsSolver(args.config)

    # evaluate or train
    if args.evaluate:
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
