from torch.utils.data import DataLoader

from .datasets import CustomDataset
from .transforms import build_transformer
from .sampler import build_sampler


def build_custom_dataloader(data_type, cfg_dataset):
    """
    arguments:
        - data_type: 'train', 'test', 'val'
        - cfg_dataset: configurations of dataset
    """
    assert data_type in cfg_dataset
    # build transformer
    transformer = build_transformer(cfg_dataset[data_type]['transforms'])
    # build dataset
    dataset = CustomDataset(
        root_dir=cfg_dataset[data_type]['root_dir'],
        meta_file=cfg_dataset[data_type]['meta_file'],
        transform=transformer,
        read_type=cfg_dataset.read_type
    )
    # add kwargs of sampler
    cfg_dataset[data_type]['sampler']['kwargs']['dataset'] = dataset
    cfg_dataset[data_type]['sampler']['kwargs']['total_iter'] = cfg_dataset['max_iter']
    cfg_dataset[data_type]['sampler']['kwargs']['last_iter'] = cfg_dataset['last_iter']
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
