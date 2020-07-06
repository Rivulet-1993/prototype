import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


class GammaCorrection(object):
    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img):
        # returned image is 3 channel with r = g = b
        img = F.to_grayscale(img, num_output_channels=3)
        img = F.adjust_gamma(img, self.gamma, self.gain)

        return img


class Cutout(object):
    def __init__(self, n_holes=2, length=32, prob=0.5):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() < self.prob:
            h = img.size(1)
            w = img.size(2)
            mask = np.ones((h, w), np.float32)
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)
                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img


transforms_info_dict = {
    'resize': transforms.Resize,
    'random_resized_crop': transforms.RandomResizedCrop,
    'random_horizontal_flip': transforms.RandomHorizontalFlip,
    'ramdom_vertical_flip': transforms.RandomVerticalFlip,
    'random_rotation': transforms.RandomRotation,
    'color_jitter': transforms.ColorJitter,
    'normalize': transforms.Normalize,
    'to_tensor': transforms.ToTensor,
    'gamma_correction': GammaCorrection,
    'cutout': Cutout
}


def build_transformer(cfgs):
    transform_list = []
    for cfg in cfgs:
        transform_type = transforms_info_dict[cfg['type']]
        kwargs = cfg['kwargs'] if cfg['kwargs'] is not None else []
        transform = transform_type(**kwargs)
        transform_list.append(transform)
    return transforms.Compose(transform_list)
