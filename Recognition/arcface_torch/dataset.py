import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import mxnet.gluon.data.vision.transforms as mx_transforms
from mxnet import nd
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn

def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None
    
    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    mx_transform = mx_transforms.Compose([
        mx_transforms.Resize((112, 112)),
        mx_transforms.ToTensor(),
        mx_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    train_dataset = BinDataset('train.rec', 'train.idx', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    return train_loader

class BinDataset(torch.utils.data.Dataset):
    def __init__(self, bin_file, idx_file, transform=None):
        self.imgrec = mx.recordio.MXIndexedRecordIO(idx_file, bin_file, 'r')
        self.samples = []
        for idx in range(self.imgrec.keys[-1] + 1):
            s = self.imgrec.read_idx(idx)
            header, img = mx.recordio.unpack(s)
            self.samples.append((img, header.label[0]))
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.samples[index]
        img = nd.array(img).astype('uint8').asnumpy()
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


class YourNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Defina sua rede neural aqui

    def forward(self, x):
        # Defina a propagação para frente da sua rede neural aqui
        return x

net = YourNetwork()


imgrec = mx.recordio.MXIndexedRecordIO('lfw.idx', 'lfw.bin', 'r')

imgidx = []
s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)
imgidx.append(0)
for idx in range(1, len(imgrec.keys)):
    s = imgrec.read_idx(idx)
    header, _ = mx.recordio.unpack(s)
    imgidx.append(idx)

with open('lfw.idx', 'wb') as f:
    np.array(imgidx, dtype=np.int64).tofile(f)

test_dataset = BinDataset('lfw.bin', 'lfw.idx', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
