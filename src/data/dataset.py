import torch
from torchvision import datasets

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import ColorJitter, RandomResizedCrop, RandomApply, RandomGrayscale, RandomHorizontalFlip
from PIL import Image
import pathlib
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from src.utils.seeds import worker_init_fn

class TransformsSimCLR:
    def __init__(self, c_w=0.5):
        color_jitter = ColorJitter(0.8*c_w, 0.8*c_w, 0.8*c_w, 0.2*c_w)
        self.transform = Compose([
            RandomResizedCrop(32, scale=(0.2, 1.)),
            RandomApply([color_jitter], p=0.8),
            RandomGrayscale(p=0.2),
            RandomHorizontalFlip(), 
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    def __call__(self, x):
        q = self.transform(x)
        k = self.transform(x)
        return [q, k]

def create_dataset(root='./data/', download=True, batch_size=256):
    ssl_transform = TransformsSimCLR()
    unsup_trainset = datasets.CIFAR100(root=root, train=True, download=download, transform=ssl_transform)
    unsup_loader = torch.utils.data.DataLoader(unsup_trainset, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)    
    return unsup_loader