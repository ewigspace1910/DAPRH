import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import math
import random

from .transform import build_transforms
from .Dataset import Market1501
from .bases import ImageDataset
#from sampler.triplet_sampler import RandomIdentitySampler

from PIL import Image
import numpy as np



def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids   #x, y


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)  #x,y, camid
    return torch.stack(imgs, dim=0), pids, camids


def make_dataloader(Cfg):
    train_transforms = build_transforms(Cfg, is_train=True)
    val_transforms = build_transforms(Cfg, is_train=False)

    num_workers = Cfg.DATALOADER_NUM_WORKERS
    dataset = Market1501(data_dir = Cfg.DATA_DIR, verbose = True)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if Cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(train_set,
            batch_size = Cfg.BATCHSIZE,
            shuffle = True,
            num_workers = num_workers,
            sampler = None, #customized batch sampler
            collate_fn = train_collate_fn, 
            drop_last = True
        )
    else:
        print('unsupported sampler! expected softmax but got {}'.format(Cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(val_set,
        batch_size=Cfg.BATCHSIZE,
        shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes