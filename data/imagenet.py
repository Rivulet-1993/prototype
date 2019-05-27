import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import ImageNetDataset
from .pipelines import ImageNetTrainPipe  # , ImageNetValPipe
from .nvidia_dali_dataloader import DaliDataLoader, dali_default_collate
from .sampler import DistributedGivenIterationSampler


def make_imagenet_train_data(config):
    """
    config fields:
        use_dali: bool
        train_root: str
        train_meta: str
        read_from: 'fake', 'mc', 'ceph', 'fs'
        batch_size: int
        dali_workers: int
        input_size: int
        augmentation:
            rotation: float
            colorjitter: bool
        max_iter: int
        last_iter: int
        workers: int
        pin_memory: bool
    """

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
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # augmentation
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

        sampler = DistributedGivenIterationSampler(
                dataset=dataset,
                total_iter=config.max_iter,
                batch_size=config.batch_size,
                last_iter=config.last_iter)

        loader = DataLoader(
                dataset, batch_size=config.batch_size, shuffle=False,
                num_workers=config.workers, pin_memory=True, sampler=sampler)

    return {'loader': loader}

# def make_imagenet_val_data():
#    # val
#    val_dataset = McDataset(
#        config.val_root,
#        config.val_source,
#        read_from=args.read_from)
#
#    val_pipeline = ImageNetValPipe(config.batch_size,
#                                    config.dali_workers,
#                                    torch.cuda.current_device(),
#                                    config.augmentation.input_size,
#                                    config.augmentation.test_resize)
#
#    val_sampler = DistributedSampler(val_dataset, round_up=False)
#
#    val_loader = DataLoader(
#        val_dataset, batch_size=config.batch_size, shuffle=False,
#        num_workers=config.workers, pin_memory=True, sampler=val_sampler,
#        collate_fn=dali_default_collate)
#
#    val_loader = DaliDataLoader(val_pipeline, dataloader=val_loader)
#
#        # val
#        val_dataset = McDataset(
#            config.val_root,
#            config.val_source,
#            transforms.Compose([
#                transforms.Resize(config.augmentation.test_resize),
#                transforms.CenterCrop(config.augmentation.input_size),
#                transforms.ToTensor(),
#                normalize,
#            ]),
#            args.fake)
