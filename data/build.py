from pathlib import Path

import torch.distributed as dist
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, \
    RandomHorizontalFlip, Resize, ToTensor

def build_loader(config): 
    dsets = dict()
    dset_loaders = dict()

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    transform = build_transform(config)
    # get the source and target dataset
    if config.model.classifier.train:
        dataset = torchvision.datasets.ImageFolder(root=config, transform=transform)

        dsets['train_source'] = torchvision.datasets.ImageFolder(root=config.data.source_data_path, transform=transform)
        print(f"local rank {config.local_rank} / global rank {dist.get_rank()} "
              f"successfully build source dataset")

        sampler_train_source = DistributedSampler(dsets['train_source'],
                                                  num_replicas=num_tasks,
                                                  rank=global_rank,
                                                  shuffle=True)

        dset_loaders['train_source'] = FFDataLoader(
            dataset=dsets['train_source'],
            sampler=sampler_train_source,
            batch_size=config.model.batch_size,
            num_workers=config.workers,
            pin_memory=config.pin_mem,
            drop_last=False,
            shuffle=False,
        )

    dsets['train'] = torchvision.datasets.ImageFolder(root=config.data.target_data_path, transform=transform)
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} "
          f"successfully build target dataset")

    sampler_train = DistributedSampler(dsets['train'],
                                       num_replicas=num_tasks,
                                       rank=global_rank,
                                       shuffle=True)

    dset_loaders['train'] = FFDataLoader(
        dataset=dsets['train'],
        sampler=sampler_train,
        batch_size=config.model.batch_size,
        num_workers=config.workers,
        pin_memory=config.pin_mem,
        drop_last=False,
        shuffle=False,
    )

    return dsets, dset_loaders



def build_dataset(config, ):
    transform = build_transform(config)
    dataset = torchvision.datasets.ImageFolder(root=config, transform=transform)
    return dataset


def build_transform(config):
    """ transform image into tensor """
    transform = Compose([
        Resize(size=[config.data.img_size, config.data.img_size], interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(p=config.aug.hflip),
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to -1, 1
    ])
    return transform
