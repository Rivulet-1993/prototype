from torch.utils.data import DataLoader

from .datasets import CustomDataset
from .transforms import build_transformer
from .sampler import build_sampler
from .metrics import build_evaluator


def build_custom_dataloader(data_type, cfg_dataset):
    """
    arguments:
        - data_type: 'train', 'test', 'val'
        - cfg_dataset: configurations of dataset
    """
    assert data_type in cfg_dataset
    # build transformer
    transformer = build_transformer(cfg_dataset[data_type]['transforms'])
    # build evaluator
    evaluator = None
    if data_type == 'test' and cfg_dataset[data_type].get('evaluator', None):
        evaluator = build_evaluator(cfg_dataset[data_type]['evaluator'])
    # build dataset
    dataset = CustomDataset(
        root_dir=cfg_dataset[data_type]['root_dir'],
        meta_file=cfg_dataset[data_type]['meta_file'],
        transform=transformer,
        read_from=cfg_dataset['read_from'],
        evaluator=evaluator,
    )
    # initialize kwargs of sampler
    cfg_dataset[data_type]['sampler']['kwargs'] = {}
    if cfg_dataset[data_type]['sampler']['type'] == 'naive':
        sampler_kwargs = {'dataset': dataset}
    else:
        sampler_kwargs = {
            'dataset': dataset,
            'batch_size': cfg_dataset['batch_size'],
            'total_iter': cfg_dataset['max_iter'],
            'last_iter': cfg_dataset['last_iter']
        }
    cfg_dataset[data_type]['sampler']['kwargs'].update(sampler_kwargs)
    # build sampler
    sampler = build_sampler(cfg_dataset[data_type]['sampler'])
    # build dataloader
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg_dataset['batch_size'],
        shuffle=False if sampler is not None else True,
        num_workers=cfg_dataset['num_workers'],
        pin_memory=cfg_dataset['pin_memory'],
        sampler=sampler
    )
    return {'type': data_type, 'loader': loader}
