from pathlib import Path
import torch
import torch.distributed as dist
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, \
    RandomHorizontalFlip, Resize, ToTensor
from PIL import Image
from .samplers import SubsetRandomSampler
import random

class CustomImageList(torch.utils.data.Dataset):
    def __init__(self, root,transform,labels = None):
        self.data_dir = Path(root)
        self.image_paths = list(self.data_dir.glob("*.jpg")) + list(self.data_dir.glob("*.png"))
        print(self.image_paths)
        self.transform = transform
        self.labels = labels  # Optional labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        print(f"Attempting to open image at: {image_path}")  # Debugging
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            label = self.labels[idx]  # Return the label if available
            return image, label
        
        return image


class FastDataLoader:
    def __init__(self, dataset, sampler=None, batch_size=32, num_workers=4, pin_memory=False, drop_last=False, shuffle=False):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        indices = random.sample(range(len(self.loader)) , min(len(self.loader),batch_size))
        self.sampler = SubsetRandomSampler(indices)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def build_loader(config):
    dsets = dict()
    dset_loaders = dict()

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    transform = build_transform(config)
    # get the source and target dataset
    if config.model.classifier.train:
        #dataset = CustomImageList(root=config.data.data_root_path, transform=transform)

        dsets['train_source'] = CustomImageList(root=config.data.source_data_path, transform=transform)

        print(f"local rank {config.local_rank} / global rank {dist.get_rank()} "
              f"successfully build source dataset")

        sampler_train_source = DistributedSampler(dsets['train_source'],
                                                  num_replicas=num_tasks,
                                                  rank=global_rank,
                                                  shuffle=True)

        dset_loaders['train_source'] = FastDataLoader(
            dataset=dsets['train_source'],
            sampler=sampler_train_source,
            batch_size=config.model.batch_size,
            num_workers=config.workers,
            pin_memory=config.pin_mem,
            drop_last=False,
            shuffle=False,
        )

    dsets['train'] = CustomImageList(root=config.data.target_data_path, transform=transform)
    print(f"local rank {config.local_rank} / global rank {dist.get_rank()} "
          f"successfully build target dataset")

    sampler_train = DistributedSampler(dsets['train'],
                                       num_replicas=num_tasks,
                                       rank=global_rank,
                                       shuffle=True)

    dset_loaders['train'] = FastDataLoader(
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
