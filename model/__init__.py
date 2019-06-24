from .mobilenet_v2 import mobilenet_v2  # noqa: F401


def model_entry(config):
    return globals()[config['type']](**config['kwargs'])
