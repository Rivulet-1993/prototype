import sys
sys.path.append('../..')
from easydict import EasyDict  # noqa: E402

import linklink as link  # noqa: E402

from data import make_imagenet_train_data  # noqa: E402
from utils.dist import dist_init  # noqa: E402

config = EasyDict()
config.use_dali = True
config.train_root = '/mnt/lustre/share/images/train'
config.train_meta = '/mnt/lustre/share/images/meta/train.txt'
config.read_from = 'mc'
config.batch_size = 64
config.dali_workers = 4
config.workers = 2
config.input_size = 224
config.augmentation = {}
config.augmentation.colorjitter = None
config.augmentation.rotation = 0
config.max_iter = 50
config.last_iter = -1
config.pin_memory = True

rank, _ = dist_init()

train_data = make_imagenet_train_data(config)
train_loader = train_data['loader']

print('num: {}'.format(len(train_loader)))

it = 0
for img, label in train_loader:
    if rank == 0:
        print('########## rank {}, iter {} ###########'.format(rank, it))
    it += 1
    if it == 10:
        break

print('done')
link.finalize()
