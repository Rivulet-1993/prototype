from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init

import linklink as link
from utils.misc import get_bn

__all__ = ['mobilenet_v2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, t, stride=1, act=nn.ReLU6):
        super(LinearBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
        self.bn1 = BN(inplanes * t)
        self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=inplanes * t)
        self.bn2 = BN(inplanes * t)
        self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
        self.bn3 = BN(outplanes)
        self.act = act(inplace=True)

        self.with_skip_connect = stride == 1 and inplanes == outplanes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.with_skip_connect:
            out += residual

        return out


class MobileNetV2(nn.Module):

    def __init__(self, scale=1.0, num_classes=1000,
                 t=[0, 1, 6, 6, 6, 6, 6, 6],
                 n=[1, 1, 2, 3, 4, 3, 3, 1],
                 c=[32, 16, 24, 32, 64, 96, 160, 320],
                 dropout=0.2,
                 bn=None):

        super(MobileNetV2, self).__init__()

        global BN
        BN = get_bn(bn)

        self.act = nn.ReLU6(inplace=True)
        self.num_classes = num_classes

        self.c = [_make_divisible(ch * scale, 8) for ch in c]
        self.s = [2, 1, 2, 2, 2, 1, 2, 1]
        self.t = t
        self.n = n
        assert self.t[0] == 0

        self.conv1 = nn.Conv2d(3, self.c[0], kernel_size=3, bias=False, stride=self.s[0], padding=1)
        self.bn1 = BN(self.c[0])

        self.main = self._make_all()

        # Last convolution has 1280 output channels for scale <= 1
        last_ch = 1280 if scale <= 1 else _make_divisible(1280 * scale, 8)
        self.conv_last = nn.Conv2d(self.c[-1], last_ch, kernel_size=1, bias=False)
        self.bn_last = BN(last_ch)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(last_ch, self.num_classes)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, link.nn.SyncBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    init.constant_(m.weight, 1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, inplanes, outplanes, n, stride, t):
        modules = OrderedDict()

        modules['MBConv_0'] = LinearBottleneck(inplanes, outplanes, t, stride=stride)

        for i in range(1, n):
            modules['MBConv_{}'.format(i)] = LinearBottleneck(outplanes, outplanes, t, stride=1)

        return nn.Sequential(modules)

    def _make_all(self):
        modules = OrderedDict()
        for i in range(1, len(self.c)):
            modules['stage_{}'.format(i)] = self._make_stage(inplanes=self.c[i-1], outplanes=self.c[i],
                                                             n=self.n[i], stride=self.s[i], t=self.t[i])
        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.main(x)

        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet_v2(**kwargs):
    model = MobileNetV2(**kwargs)
    return model
