from .mobilenet_v2 import mobilenet_v2  # noqa: F401
from .regnet import (  # noqa: F401
    regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
)
from .resnet import (  # noqa: F401
    resnet18, resnet26, resnet34, resnet50,
    resnet101, resnet152, resnet_custom
)
from .preact_resnet import (  # noqa: F401
    preact_resnet18, preact_resnet34, preact_resnet50,
    preact_resnet101, preact_resnet152
)
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
from .nas_zoo import (  # noqa: F401
    mbnas_t29_x0_84, mbnas_t47_x1_00, supnas_t18_x1_00, supnas_t37_x0_92, supnas_t44_x1_00,
    supnas_t66_x1_11, supnas_t100_x0_96, nas_custom
)
from .nas_backbone import nas_backbone  # noqa: F401
from .resnet_official import (  # noqa: F401
    resnet18_official, resnet34_official, resnet50_official, resnet101_official, resnet152_official,
    resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
)
from .condconv_resnet import (  # noqa: F401
    resnet18_condconv_shared, resnet18_condconv_independent,
    resnet34_condconv_shared, resnet34_condconv_independent,
    resnet50_condconv_shared, resnet50_condconv_independent,
    resnet101_condconv_shared, resnet101_condconv_independent,
    resnet152_condconv_shared, resnet152_condconv_independent
)
from .condconv_mobilenet_v2 import (  # noqa: F401
    mobilenetv2_condconv_pointwise, mobilenetv2_condconv_independent, mobilenetv2_condconv_shared
)
from .mobilenet_v3 import mobilenet_v3  # noqa: F401
from .ghostnet import ghostnet  # noqa: F401
from .resnest import resnest50, resnest101, resnest200, resnest269  # noqa: F401
from .ibnnet import resnet50_ibn_a, resnet101_ibn_a, resnet152_ibn_a  # noqa: F401
from .fbnet_v2 import fbnetv2_f1, fbnetv2_f4, fbnetv2_l2_hs, fbnetv2_l3  # noqa: F401
from .weightnet_shufflenet_v2 import (  # noqa: F401
    shufflenet_v2_x0_5_weightnet, shufflenet_v2_x1_0_weightnet,
    shufflenet_v2_x1_5_weightnet, shufflenet_v2_x2_0_weightnet
)
from .mobilenext import mobilenext  # noqa: F401
from .dmcp_resnet import (  # noqa: F401
    dmcp_resnet18_45M, dmcp_resnet18_47M, dmcp_resnet18_51M, dmcp_resnet18_480M, dmcp_resnet50_282M,
    dmcp_resnet50_1100M, dmcp_resnet50_2200M
)
from .bignas_resnet_basicblock import (  # noqa: F401
    bignas_resnet18_9M, bignas_resnet18_37M, bignas_resnet18_50M,
    bignas_resnet18_49M, bignas_resnet18_66M, bignas_resnet18_1555M,
    bignas_resnet18_107M, bignas_resnet18_125M, bignas_resnet18_150M,
    bignas_resnet18_312M, bignas_resnet18_403M, bignas_resnet18_492M
)
from .bignas_resnet_bottleneck import (  # noqa: F401
    bignas_resnet50_2954M, bignas_resnet50_3145M, bignas_resnet50_3811M
)


def model_entry(config):

    if config['type'] not in globals():
        from prototype.spring import PrototypeHelper
        return PrototypeHelper.external_model_builder[config['type']](**config['kwargs'])

    return globals()[config['type']](**config['kwargs'])
