from PIL import Image
import json
import io
import os.path as osp

from .base_dataset import BaseDataset


def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        print('Failed in loading {}'.format(filepath))
    return img


class CustomDataset(BaseDataset):
    """
    Custom Dataset using self-defined setting.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_type (:obj:`str`): read type from the original meta_file

    Example of meta_file in json type:
        - {'filename': 'n01440764/n01440764_10026.JPEG', 'label': 0, 'label_name': 'dog'}

    """
    def __init__(self,
                 root_dir,
                 meta_file,
                 transform=None,
                 read_type='mc'):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.read_type = read_type

        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        self.labels = []
        for line in lines:
            info = json.loads(line)
            self.metas.append((info['filename'], info['label']))
            self.labels.append(int(info['label']))

        super(CustomDataset, self).__init__(read_from=read_type)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        label = self.metas[idx][1]
        filename = osp.join(self.root_dir, self.metas[idx][0])
        img_bytes = self.read_file(filename)
        img = pil_loader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'label': label, 'filename': filename}
