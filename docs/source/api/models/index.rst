prototype.model
===============================

.. toctree::
   :maxdepth: 2

ResNet
----------

.. autoclass:: prototype.model.resnet.ResNet
    :members: __init__, forward

.. autofunction:: prototype.model.resnet18

.. autofunction:: prototype.model.resnet26

.. autofunction:: prototype.model.resnet34

.. autofunction:: prototype.model.resnet50

.. autofunction:: prototype.model.resnet101

.. autofunction:: prototype.model.resnet152

.. autofunction:: prototype.model.resnet_custom

ResNeXt
--------------------

.. autofunction:: prototype.model.resnext50_32x4d

.. autofunction:: prototype.model.resnext101_32x8d

Wide-ResNet
--------------------

.. autofunction:: prototype.model.wide_resnet50_2

.. autofunction:: prototype.model.wide_resnet101_2

PreactResNet
--------------------

.. autoclass:: prototype.model.preact_resnet.PreactResNet
    :members: __init__, forward

.. autofunction:: prototype.model.preact_resnet18

.. autofunction:: prototype.model.preact_resnet34

.. autofunction:: prototype.model.preact_resnet50

.. autofunction:: prototype.model.preact_resnet101

.. autofunction:: prototype.model.preact_resnet152


DenseNet
--------------------

.. autoclass:: prototype.model.densenet.DenseNet
    :members: __init__, forward

.. autofunction:: prototype.model.densenet121

.. autofunction:: prototype.model.densenet169

.. autofunction:: prototype.model.densenet201

.. autofunction:: prototype.model.densenet161

EfficientNet
------------

.. autoclass:: prototype.model.efficientnet.EfficientNet
    :members: __init__, forward

.. autofunction:: prototype.model.efficientnet_b0

.. autofunction:: prototype.model.efficientnet_b1

.. autofunction:: prototype.model.efficientnet_b2

.. autofunction:: prototype.model.efficientnet_b3

.. autofunction:: prototype.model.efficientnet_b4

.. autofunction:: prototype.model.efficientnet_b5

.. autofunction:: prototype.model.efficientnet_b6

.. autofunction:: prototype.model.efficientnet_b7

RegNet
------------

.. autoclass:: prototype.model.regnet.RegNet
    :members: __init__, forward

.. autofunction:: prototype.model.regnetx_200m

.. autofunction:: prototype.model.regnetx_400m

.. autofunction:: prototype.model.regnetx_600m

.. autofunction:: prototype.model.regnetx_800m

.. autofunction:: prototype.model.regnetx_1600m

.. autofunction:: prototype.model.regnetx_3200m

.. autofunction:: prototype.model.regnetx_4000m

.. autofunction:: prototype.model.regnetx_6400m

.. autofunction:: prototype.model.regnety_200m

.. autofunction:: prototype.model.regnety_400m

.. autofunction:: prototype.model.regnety_600m

.. autofunction:: prototype.model.regnety_800m

.. autofunction:: prototype.model.regnety_1600m

.. autofunction:: prototype.model.regnety_3200m

.. autofunction:: prototype.model.regnety_4000m

.. autofunction:: prototype.model.regnety_6400m

HRNet
------------

.. autoclass:: prototype.model.hrnet.HighResolutionNet
    :members: __init__, forward

.. autofunction:: prototype.model.HRNet

Kwargs of HRNet-w18 is given in `"w18-config.yaml" <http://gitlab.bj.sensetime.com/spring-ce/element/prototype/blob/master/model_zoo_exp/hrnet_w18_batch1k_epoch100_steplr_nesterov_wd0.0001/config.yaml>`_

Kwargs of HRNet-w30 is given in `"w30-config.yaml" <http://gitlab.bj.sensetime.com/spring-ce/element/prototype/blob/master/model_zoo_exp/hrnet_w30_batch1k_epoch100_steplr_nesterov_wd0.0001/config.yaml>`_

Kwargs of HRNet-w32 is given in `"w32-config.yaml" <http://gitlab.bj.sensetime.com/spring-ce/element/prototype/blob/master/model_zoo_exp/hrnet_w32_batch1k_epoch100_steplr_nesterov_wd0.0001/config.yaml>`_

Kwargs of HRNet-w40 is given in `"w-40config.yaml" <http://gitlab.bj.sensetime.com/spring-ce/element/prototype/blob/master/model_zoo_exp/hrnet_w40_batch1k_epoch100_steplr_nesterov_wd0.0001/config.yaml>`_

Kwargs of HRNet-w44 is given in `"w-44config.yaml" <http://gitlab.bj.sensetime.com/spring-ce/element/prototype/blob/master/model_zoo_exp/hrnet_w44_batch1k_epoch100_steplr_nesterov_wd0.0001/config.yaml>`_

Kwargs of HRNet-w48 is given in `"w-48config.yaml" <http://gitlab.bj.sensetime.com/spring-ce/element/prototype/blob/master/model_zoo_exp/hrnet_w48_batch1k_epoch100_steplr_nesterov_wd0.0001/config.yaml>`_

Kwargs of HRNet-w60 is given in `"w-60config.yaml" <http://gitlab.bj.sensetime.com/spring-ce/element/prototype/blob/master/model_zoo_exp/hrnet_w60_batch1k_epoch100_steplr_nesterov_wd0.0001/config.yaml>`_

Kwargs of HRNet-w72 is given in `"w-72config.yaml" <http://gitlab.bj.sensetime.com/spring-ce/element/prototype/blob/master/model_zoo_exp/hrnet_w72_batch1k_epoch100_steplr_nesterov_wd0.0001/config.yaml>`_

MobileNet-V2
----------------

.. autoclass:: prototype.model.mobilenet_v2.MobileNetV2
    :members: __init__, forward

.. autofunction:: prototype.model.mobilenet_v2

MobileNet-V3
----------------

.. autoclass:: prototype.model.mobilenet_v3.MobileNetV3
    :members: __init__, forward

.. autofunction:: prototype.model.mobilenet_v3

ShuffleNet-V2
----------------

.. autoclass:: prototype.model.shufflenet_v2.ShuffleNetV2
    :members: __init__, forward

.. autofunction:: prototype.model.shufflenet_v2_x0_5

.. autofunction:: prototype.model.shufflenet_v2_x1_0

.. autofunction:: prototype.model.shufflenet_v2_x1_5

.. autofunction:: prototype.model.shufflenet_v2_x2_0

.. autofunction:: prototype.model.shufflenet_v2_scale

MNASNet
----------------

.. autoclass:: prototype.model.mnasnet.MNASNet
    :members: __init__, forward

.. autofunction:: prototype.model.mnasnet

SupNAS
----------------

.. autoclass:: prototype.model.nas_zoo.ValImageNet
    :members: __init__, forward

.. autofunction:: prototype.model.mbnas_t29_x0_84

.. autofunction:: prototype.model.mbnas_t47_x1_00

.. autofunction:: prototype.model.supnas_t18_x1_00

.. autofunction:: prototype.model.supnas_t37_x0_92

.. autofunction:: prototype.model.supnas_t44_x1_00

.. autofunction:: prototype.model.supnas_t66_x1_11

.. autofunction:: prototype.model.supnas_t100_x0_96

SENet
----------------

.. autoclass:: prototype.model.senet.SENet
    :members: __init__, forward

.. autofunction:: prototype.model.se_resnext50_32x4d

.. autofunction:: prototype.model.se_resnext101_32x4d

ResNeSt
----------------

.. autoclass:: prototype.model.resnest.ResNeSt
    :members: __init__, forward

.. autofunction:: prototype.model.resnest50

.. autofunction:: prototype.model.resnest101

.. autofunction:: prototype.model.resnest200

.. autofunction:: prototype.model.resnest269
