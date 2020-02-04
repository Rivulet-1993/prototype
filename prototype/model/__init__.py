from .mobilenet_v2 import mobilenet_v2  # noqa: F401
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152  # noqa: F401
from .efficientnet import (  # noqa: F401
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
)
from .shufflenet_v2 import (  # noqa: F401
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0, shufflenet_v2_scale
)
from .senet import se_resnext50_32x4d, se_resnext101_32x4d  # noqa: F401
from .densenet import densenet121, densenet169, densenet201, densenet161  # noqa: F401
# from .toponet import toponet_conv, toponet_sepconv, toponet_mb
from .hrnet import HRNet  # noqa: F401
from .mnasnet import mnasnet  # noqa: F401
from .nas_zoo import mbnas_t29_x0_84, mbnas_t47_x1_00  # noqa: F401
from .resnet_official import (  # noqa: F401
    resnet18_official, resnet34_official, resnet50_official, resnet101_official, resnet152_official,
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
)
from .mobilenet_v3 import mobilenet_v3  # noqa: F401


def model_entry(config):
    return globals()[config['type']](**config['kwargs'])
