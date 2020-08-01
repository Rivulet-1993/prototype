import torch
import torch.nn as nn


class NASBackbone(nn.Module):
    def __init__(self, **kwargs):
        super(NASBackbone, self).__init__()
        num_classes = kwargs.get('num_classes', 1000)
        assert 'backbone_type' in kwargs
        backbone_type = kwargs.pop('backbone_type')
        kwargs['output_config'] = {'type': 'index', 'output_feature_idx': [-1]}
        backbone_config = {'type': backbone_type, 'kwargs': kwargs}
        self.backbone = self.backbone_entry(backbone_config)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.backbone.get_output_channels()[-1],
                            num_classes)

    def backbone_entry(self, config):
        if config['type'] not in globals():
            from prototype.spring import PrototypeHelper
            return PrototypeHelper.external_model_builder[
                config['type']](**config['kwargs'])
        return globals()[config['type']](**config['kwargs'])

    def forward(self, x):
        x = self.backbone(x)[0]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def nas_backbone(**kwargs):
    return NASBackbone(**kwargs)
