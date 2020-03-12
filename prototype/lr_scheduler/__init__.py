from .scheduler import Step, StepDecay, Cosine, Poly # noqa F401


def scheduler_entry(config):

    return globals()[config.type](**config.kwargs)
