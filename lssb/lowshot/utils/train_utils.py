import torch
import os
import pdb
import pytorch_lightning as pl
import numpy as np

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models
from argparse import ArgumentParser

import lssb.nets as nets
from lssb.data.sampler import CategoriesSampler
from lssb.data.modelnet import ModelNet
from lssb.data.toys import Toys4K
from lssb.data.shapenet import ShapeNet55
from lssb.data.mImNetLoader import DatasetFolder

def get_dataset(dataset, modality, use_aug, extra_args=None):
    
    if dataset == 'modelnet':
        train_dataset = ModelNet(
            split='train', 
            modality=modality,
            use_aug=use_aug,
            extra_args=extra_args
        )
        
        val_dataset = ModelNet(
            split='val', 
            modality=modality,
            use_aug=False,
            extra_args=extra_args
        )
       
        test_dataset = ModelNet(
            split='test', 
            modality=modality,
            use_aug=False,
            extra_args=extra_args
        )
     
    if 'shapenet' in dataset:

        train_dataset = ShapeNet55(
            split='train', 
            modality=modality,
            use_aug=use_aug,
            extra_args=extra_args
        )
        
        val_dataset = ShapeNet55(
            split='val', 
            modality=modality,
            use_aug=False,
            extra_args=extra_args
        )
       
        test_dataset = ShapeNet55(
            split='test', 
            modality=modality,
            use_aug=False,
            extra_args=extra_args
        )
   
    if 'toys' in dataset:
        train_dataset = Toys4K(
            split='train', 
            modality=modality,
            use_aug=use_aug,
            extra_args=extra_args
        )
        
        val_dataset = Toys4K(
            split='val', 
            modality=modality,
            use_aug=False,
            extra_args=extra_args
        )
       
        test_dataset = Toys4K(
            split='test', 
            modality=modality,
            use_aug=False,
            extra_args=extra_args
        )


    if dataset == 'mini-imagenet':
        if modality != 'image':
            raise Exception("mini imagenet is an image only dataset")

        train_dataset = DatasetFolder('train', use_aug)

        val_dataset = DatasetFolder('val', False)

        test_dataset = DatasetFolder('test', False) 

    print("# training samples:", len(train_dataset), train_dataset.use_aug)
    print("# validation samples:", len(val_dataset), val_dataset.use_aug)
    print("# test samples:", len(test_dataset), test_dataset.use_aug)
    
    return train_dataset, val_dataset, test_dataset
