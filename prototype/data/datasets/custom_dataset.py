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
    Custom Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_from (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics

    Metafile example::
        "{'filename': 'n01440764/n01440764_10026.JPEG', 'label': 0, 'label_name': 'dog'}\n"
    """
    def __init__(self, root_dir, meta_file, transform=None, read_from='mc', evaluator=None):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.initialized = False

        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            info = json.loads(line)
            self.metas.append((info['filename'], int(info['label']), info['label_name']))

        super(CustomDataset, self).__init__(root_dir=root_dir,
                                            meta_file=meta_file,
                                            read_from=read_from,
                                            transform=transform,
                                            evaluator=evaluator)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = osp.join(self.root_dir, self.metas[idx][0])
        label = self.metas[idx][1]
        label_name = self.metas[idx][2]
        img_bytes = self.read_file(filename)
        img = pil_loader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        item = {
            'image': img,
            'label': label,
            'image_id': idx,
            'filename': filename,
            'label_name': label_name
        }
        return item

    def dump(self, writer, output):
        filename = output['filename']
        image_id = output['image_id']
        label_name = output['label_name']
        prediction = self.tensor2numpy(output['prediction'])
        score = self.tensor2numpy(output['score'])
        label = self.tensor2numpy(output['label'])
        for _idx in range(len(filename)):
            res = {
                'filename': filename[_idx],
                'image_id': int(image_id[_idx]),
                'label_name': label_name[_idx],
                'prediction': int(prediction[_idx]),
                'score': [float('%.8f' % s) for s in score[_idx]],
                'label': int(label[_idx])
            }
            writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()
