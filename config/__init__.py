import yaml
from easydict import EasyDict
from .imagenet import imagenet  # noqa: F401


def parse_config(config_file):

    with open(config_file) as f:
        config = yaml.load(f)

    config = EasyDict(config)

    return globals()[config.type](config.config)
