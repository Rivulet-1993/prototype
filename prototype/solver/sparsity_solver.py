import argparse
from .cls_solver import ClsSolver
from prototype.core import ASP
from prototype.optimizer import optim_entry
from prototype.utils.ema import EMA
from prototype.utils.dist import link_dist
from prototype.utils.misc import param_group_all, load_state_optimizer


class SparsitySolver(ClsSolver):

    def __init__(self, config_file):
        super().__init__(config_file)
        # assert 'optimizer' in self.state and 'model' in self.state, 'Must load pretrained model!'
        ASP.prune_trained_model(self.model, self.optimizer)
        assert ASP.is_sparsity_enabled(), 'ASP is not able, something is wrong!'
        # build ema after asp
        self.build_ema()

    def build_optimizer(self):
        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type
        # make param_groups
        pconfig = {}
        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        param_group, type2num = param_group_all(self.model, pconfig)
        opt_config.kwargs.params = param_group
        self.optimizer = optim_entry(opt_config)
        if 'optimizer' in self.state:
            load_state_optimizer(self.optimizer, self.state['optimizer'])

    def build_ema(self):
        if self.config.ema.enable:
            self.config.ema.kwargs.model = self.model
            self.ema = EMA(**self.config.ema.kwargs)
        else:
            self.ema = None

        if 'ema' in self.state:
            self.ema.load_state_dict(self.state['ema'])


@link_dist
def main():
    parser = argparse.ArgumentParser(description='sparsity solver')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    # build solver
    solver = SparsitySolver(args.config)
    if solver.config.data.last_iter < solver.config.data.max_iter:
        solver.train()
    else:
        solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
