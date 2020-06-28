import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import ImageNetDataset
from .pipelines import ImageNetTrainPipe, ImageNetValPipe, ImageNetTrainPipeV2, ImageNetValPipeV2
from .nvidia_dali_dataloader import DaliDataLoader, dali_default_collate
from .sampler import DistributedGivenIterationSampler, DistributedEpochSampler, DistributedSampler
from .autoaugment import ImageNetPolicy, Cutout  # noqa: F401
from .moco_transform import TwoCropsTransform, GaussianBlur
from prototype.utils.misc import get_logger

try:
    import linklink.dali as link_dali
except ModuleNotFoundError:
    print('import linklink.dali failed, linklink version should >= 0.2.0')


def make_imagenet_train_data(config):
    """
    config fields:
        use_dali: bool
        read_from: 'fake', 'mc', 'ceph', 'fs'
        batch_size: int
        dali_workers: int
        workers: int
        pin_memory: bool
        epoch_wise_shuffle: bool
        input_size: int
        test_resize: int
        augmentation:
            rotation: float
            colorjitter: bool
        train_root: str
        train_meta: str
        val_root: str
        val_meta: str

        max_iter: int
        last_iter: int
    """

    logger = get_logger(__name__)

    config.moco_protocol = config.get('moco_protocol', False)
    config.linear_protocol = config.get('linear_protocol', False)

    if config.use_dali:
        dataset = ImageNetDataset(
            config.train_root,
            config.train_meta,
            read_from=config.read_from)

        pipeline = ImageNetTrainPipe(config.batch_size,
                                     config.dali_workers,
                                     torch.cuda.current_device(),
                                     config.input_size,
                                     colorjitter=config.augmentation.colorjitter)

        if config.epoch_wise_shuffle:
            sampler = DistributedEpochSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)

        torch_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=config.pin_memory, sampler=sampler,
            collate_fn=dali_default_collate)

        loader = DaliDataLoader(pipeline, dataloader=torch_loader)

    elif config.use_dali_v2:
        dataset = ImageNetDataset(
            config.train_root,
            config.train_meta,
            read_from=config.read_from)

        if config.epoch_wise_shuffle:
            sampler = DistributedEpochSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)

        pipeline = ImageNetTrainPipeV2(config.train_root,
                                       config.train_meta,
                                       sampler,
                                       config.input_size,
                                       colorjitter=config.augmentation.colorjitter)

        loader = link_dali.DataLoader(pipeline, config.batch_size, len(sampler),
                                      config.dali_workers,
                                      last_iter=config.last_iter)

    elif config.moco_protocol:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if config.augmentation.type == "moco_v1":
            aug = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        elif config.augmentation.type == "moco_v2" or config.augmentation.type == 'simclr':
            aug = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
            raise RuntimeError("undefined augmentation type for unsupervised learning")

        dataset = ImageNetDataset(
            config.train_root,
            config.train_meta,
            TwoCropsTransform(transforms.Compose(aug)),
            read_from=config.read_from)

        if config.epoch_wise_shuffle:
            sampler = DistributedEpochSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)

        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True, sampler=sampler)

    elif config.linear_protocol:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        aug = [
            transforms.RandomResizedCrop(config.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

        dataset = ImageNetDataset(
            config.train_root,
            config.train_meta,
            transforms.Compose(aug),
            read_from=config.read_from)

        if config.epoch_wise_shuffle:
            sampler = DistributedEpochSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)

        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True, sampler=sampler)

    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # augmentation
        if config.get('autoaugment', False):
            logger.info('Use Auto-Augmentation of ImageNetPolicy!')
            aug = [transforms.RandomResizedCrop(config.input_size),
                   ImageNetPolicy()]
        else:
            aug = [transforms.RandomResizedCrop(config.input_size),
                   transforms.RandomHorizontalFlip()]

            for k in config.augmentation.keys():
                assert k in ['rotation', 'colorjitter']
            rotation = config.augmentation.get('rotation', 0)
            colorjitter = config.augmentation.get('colorjitter', None)

            if rotation > 0:
                aug.append(transforms.RandomRotation(rotation))

            if colorjitter is not None:
                aug.append(transforms.ColorJitter(*colorjitter))

        aug.append(transforms.ToTensor())
        aug.append(normalize)

        dataset = ImageNetDataset(
            config.train_root,
            config.train_meta,
            transforms.Compose(aug),
            read_from=config.read_from)

        if config.epoch_wise_shuffle:
            sampler = DistributedEpochSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)

        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=True, sampler=sampler)

    return {'loader': loader}


def make_imagenet_val_data(config, periodic=False):
    """Generate validation dataloader for evaluation or NAS.

    args:
        config (EasyDict): configuration for building the dataset.
        periodic (bool): default ``False``,
            If ``True``, samples will be called periodically (for neural network search);
            If ``False``, samples will be called once for evaluation.
    """

    if config.use_dali:
        dataset = ImageNetDataset(
            config.val_root,
            config.val_meta,
            read_from=config.read_from)

        pipeline = ImageNetValPipe(config.batch_size,
                                   config.dali_workers,
                                   torch.cuda.current_device(),
                                   config.input_size,
                                   config.test_resize)

        if periodic:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedSampler(dataset, round_up=False)

        torch_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=config.pin_memory, sampler=sampler,
            collate_fn=dali_default_collate)

        loader = DaliDataLoader(pipeline, dataloader=torch_loader)

    elif config.use_dali_v2:

        dataset = ImageNetDataset(
            config.val_root,
            config.val_meta,
            read_from=config.read_from)

        if periodic:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedSampler(dataset, round_up=False)

        pipeline = ImageNetValPipeV2(config.val_root,
                                     config.val_meta,
                                     sampler,
                                     config.input_size,
                                     config.test_resize)

        loader = link_dali.DataLoader(
            pipeline, config.batch_size, len(sampler), config.dali_workers)

    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        dataset = ImageNetDataset(
            config.val_root,
            config.val_meta,
            transforms.Compose([
                transforms.Resize(config.test_resize),
                transforms.CenterCrop(config.input_size),
                transforms.ToTensor(),
                normalize,
            ]),
            read_from=config.read_from)

        if periodic:
            sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)
        else:
            sampler = DistributedSampler(dataset, round_up=False)

        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.workers, pin_memory=config.pin_memory, sampler=sampler)

    return {'loader': loader}
