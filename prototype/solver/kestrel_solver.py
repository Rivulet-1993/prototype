import os
import argparse
from easydict import EasyDict
import shutil
import numpy as np
import yaml
import json
import linklink as link

from prototype.utils.dist import link_dist
from .cls_solver import ClsSolver
from prototype.config import parse_config


class KestrelSolver(ClsSolver):

    def __init__(self, config_file, recover=''):
        self.config_file = config_file
        self.recover = recover
        self.config = parse_config(config_file)
        self.setup_env()
        self.build_model()

    def to_caffe(self, save_prefix='model', input_size=None):
        try:
            from spring.nart.tools import pytorch
        except ImportError:
            print('Install Spring NART first!')

        with pytorch.convert_mode():
            pytorch.convert(self.model.float(),
                            [(3, self.config.data.input_size,
                              self.config.data.input_size)],
                            filename=save_prefix,
                            input_names=['data'],
                            output_names=['out'])

    def refactor_config(self):
        '''Prepare configuration for kestrel classifier model. For details:
        https://confluence.sensetime.com/display/VIBT/nart.tools.kestrel.classifier
        '''
        kestrel_config = EasyDict()

        if hasattr(self.config, 'pixel_means'):
            kestrel_config['pixel_means'] = self.config['pixel_means']
        else:
            kestrel_config['pixel_means'] = [124.16, 116.736, 103.936]

        if hasattr(self.config, 'pixel_stds'):
            kestrel_config['pixel_stds'] = self.config['pixel_stds']
        else:
            kestrel_config['pixel_stds'] = [58.624, 57.344, 57.6]

        kestrel_config['is_rgb'] = self.config.get('is_rgb', True)
        kestrel_config['save_all_label'] = self.config.get(
            'save_all_label', True)
        kestrel_config['type'] = self.config.get('type', 'ImageNet')

        if hasattr(self.config, 'class_label'):
            kestrel_config['class_label'] = self.config['class_label']
        else:
            kestrel_config['class_label'] = {}
            kestrel_config['class_label']['imagenet'] = {}
            kestrel_config['class_label']['imagenet']['calculator'] = 'bypass'
            num_classes = self.config.model.get('num_classes', 1000)
            kestrel_config['class_label']['imagenet']['labels'] = [
                str(i) for i in np.arange(num_classes)]
            kestrel_config['class_label']['imagenet']['feature_start'] = 0
            kestrel_config['class_label']['imagenet']['feature_end'] = num_classes - 1

        self.kestrel_config = kestrel_config

    def to_kestrel(self, save_to=None):
        prefix = 'model'
        self.logger.info('Converting Model to Caffe...')
        if self.dist.rank == 0:
            self.to_caffe(prefix)

        link.synchronize()
        self.logger.info('To Caffe Done!')

        prototxt = '{}.prototxt'.format(prefix)
        caffemodel = '{}.caffemodel'.format(prefix)
        version = '1.0.0'
        model_name = self.config.model.type

        kestrel_model = '{}_{}.tar'.format(model_name, version)
        to_kestrel_yml = 'temp_to_kestrel.yml'
        self.refactor_config()

        with open(to_kestrel_yml, 'w') as f:
            yaml.dump(json.loads(json.dumps(self.kestrel_config)), f)

        cmd = 'python -m spring.nart.tools.kestrel.classifier {} {} -v {} -c {} -n {}'.format(
            prototxt, caffemodel, version, to_kestrel_yml, model_name)

        self.logger.info('Converting Model to Kestrel...')
        if self.dist.rank == 0:
            os.system(cmd)

        link.synchronize()
        self.logger.info('To Kestrel Done!')

        if save_to is None:
            save_to = kestrel_model
        else:
            save_to = os.path.join(save_to, kestrel_model)

        shutil.move(kestrel_model, save_to)
        link.synchronize()
        self.logger.info('Save kestrel model to: {}'.format(save_to))


@link_dist
def main():
    parser = argparse.ArgumentParser(description='caffe/kestrel solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--recover', type=str, default='')

    args = parser.parse_args()

    # build or recover solver
    solver = KestrelSolver(args.config, recover=args.recover)

    # to caffe and to kestrel
    solver.to_kestrel()


if __name__ == '__main__':
    main()
