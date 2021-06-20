import torch
import argparse
import numpy as np
import os

from lssb.lowshot.models.ptcld_simpleshot_classifier import PtcldClassifier
from lssb.data.modelnet import ModelNet
from lssb.feat_extract.utils import get_multiple_features
from torch.utils.data import DataLoader


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path',
                        metavar='ckpt_path',
                        type=str)

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt_path)


    model = PtcldClassifier.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.net = model.net.cuda()

    extra_args = {'use_random_SO3_rotation':False,
                  'num_points':1024}

    batch_size=64

    train_dataset = ModelNet(
        split='train', 
        modality='ptcld',
        use_aug=False,
        extra_args=extra_args
    )

    val_dataset = ModelNet(
        split='val', 
        modality='ptcld',
        use_aug=False,
        extra_args=extra_args
    )

    test_dataset = ModelNet(
        split='test', 
        modality='ptcld',
        use_aug=False,
        extra_args=extra_args
    )

    train_loader = DataLoader(
            train_dataset, 
            shuffle=False, 
            num_workers=5, 
            batch_size=batch_size, 
            drop_last=False)

    val_loader = DataLoader(
            val_dataset, 
            shuffle=False, 
            num_workers=5, 
            batch_size=batch_size, 
            drop_last=False)

    test_loader = DataLoader(
            test_dataset, 
            shuffle=False, 
            num_workers=5, 
            batch_size=batch_size, 
            drop_last=False)
    
    print("Feature extraction for training set")
    train = get_multiple_features(model, train_loader, 10)
    print("Feature extraction for validation set")
    val = get_multiple_features(model, val_loader, 10)
    print("Feature extraction for testing set")
    test = get_multiple_features(model, test_loader, 10)
    
    output_dict = {}

    output_dict['train'] = train
    output_dict['val'] = val
    output_dict['test'] = test
    
    out_dir = 'modelnet_features'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.savez(os.path.join(out_dir, 'dgcnn_ptcld_feat_dict.npz'), 
             feat_dict=output_dict)

if __name__ == "__main__":
    main()
