from .imagenet_evaluator import ImageNetEvaluator
from .custom_evaluator import CustomEvaluator


def build_evaluator(cfg):
    evaluator = {
        'custom': CustomEvaluator,
        'imagenet': ImageNetEvaluator,
    }[cfg['type']]
    return evaluator(**cfg['kwargs'])
