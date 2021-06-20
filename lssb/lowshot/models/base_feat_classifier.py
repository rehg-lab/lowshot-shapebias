import torch
import os
import pdb
import pytorch_lightning as pl
import numpy as np

from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

import lssb.nets as nets
from lssb.data.sampler import CategoriesSampler
from lssb.lowshot.utils.train_utils import get_dataset

from lssb.lowshot.utils.feat_utils import count_acc
from lssb.lowshot.utils.simpleshot_utils import compute_confidence_interval

#data loading

class FeatBase(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        
        self.hparams = hparams
        self.extra_args = None
        self.balance = self.hparams.balance
        self._build_model()
        
    def _build_model(self):
        """
            to be implemented by either "image_feat_classifier"
            or joint_feat_classifier" subclasses
        """

        raise NotImplementedError
   
    def forward(self, x, n_ways, n_shots, n_queries):
        """
            to be implemented by either "image_feat_classifier" 
            or "joint_feat_classifier" subclasses
        """
        raise NotImplementedError

    def prepare_label(self, n_ways, n_shots, n_queries):

         label = torch.arange(n_ways, dtype=torch.int16).repeat(n_queries)
         label_aux = torch.arange(n_ways, dtype=torch.int8).repeat(n_shots + n_queries)

         label = label.type(torch.LongTensor)
         label_aux = label_aux.type(torch.LongTensor)

         if torch.cuda.is_available():
             label = label.cuda()
             label_aux = label_aux.cuda()

         return label, label_aux
    
    def split_instances(self, n_ways, n_shots, n_queries):
        """
            borrowed from original feat repo 
            https://github.com/Sha-Lab/FEAT/blob/47bdc7c1672e00b027c67469d0291e7502918950/model/models/base.py#L27

        """

        return  (
                        torch.Tensor(
                            np.arange(n_ways*n_shots)
                        ).long().view(1, n_shots, n_ways)
                    , \
                        torch.Tensor(
                            np.arange(n_ways*n_shots, \
                            n_ways * (n_shots + n_queries))
                        ).long().view(1, n_queries, n_ways)
                )

    def training_step(self, batch, batch_nb):  
        output = self(
                batch,
                self.hparams.n_ls_train_ways,
                self.hparams.n_ls_train_shots,
                self.hparams.n_ls_train_queries
            )

        return output

    def training_step_end(self, output):

        logits, reg_logits = output    
        labels, labels_aux = self.prepare_label(
                self.hparams.n_ls_train_ways,
                self.hparams.n_ls_train_shots, 
                self.hparams.n_ls_train_queries
            )
        
        loss = F.cross_entropy(logits, labels)
        total_loss = loss + self.balance * F.cross_entropy(reg_logits,
                labels_aux)
               
        tensorboard_logs = {'train/batch_loss': total_loss}
        correct = (torch.argmax(logits, dim=1) == labels).float()

        return {'loss': total_loss, 
                'log':tensorboard_logs, 
                'correct':correct}
    
    def validation_step(self, batch, batch_nb, loader_idx):
            
        logits = self(
                batch,
                self.hparams.n_ls_val_ways,
                self.hparams.n_ls_val_shots,
                self.hparams.n_ls_val_queries
            )

        labels, _ = self.prepare_label(
                self.hparams.n_ls_val_ways,
                self.hparams.n_ls_val_shots, 
                self.hparams.n_ls_val_queries
            )
        
        acc = count_acc(logits, labels)
        
        return {'episode_acc': acc}

    def test_step(self, batch, batch_idx):
        logits = self(batch,
                self.hparams.n_ways,
                self.hparams.n_shots,
                self.hparams.n_ls_test_queries
            )

        labels, _ = self.prepare_label(
                self.hparams.n_ways,
                self.hparams.n_shots, 
                self.hparams.n_ls_test_queries
            )
        
        logits = logits.cuda()
        acc = count_acc(logits, labels)
        
        return {'episode_acc': acc}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = torch.cat([x['correct'] for x in outputs])
        avg_episode_acc = correct.mean()
        
        
        tensorboard_logs = {'train/loss_epoch': avg_loss,
                            'train/avg_episode_acc': avg_episode_acc}
        
        return {'train_loss': avg_loss, 'log': tensorboard_logs}
   
    def validation_epoch_end(self, outputs):
        
        val_accs = torch.Tensor([x['episode_acc'] for x in outputs[0]])
        val_ls_acc = val_accs.mean()
        
        train_accs = torch.Tensor([x['episode_acc'] for x in outputs[1]])
        train_ls_acc = train_accs.mean()

        tensorboard_logs = {'val/lowshot_acc':val_ls_acc, 
                            'val/train_lowshot_acc':train_ls_acc}
        
        return {'val_acc': val_ls_acc, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        accuracies = np.array([x['episode_acc'] for x in outputs])
        m, pm = compute_confidence_interval(accuracies)

        out_str = "{} Shot {} Way {:.4f}({:.4f})\n\n".format(
                self.hparams.n_shots, 
                self.hparams.n_ways, 
                m,
                pm)
    
        out_fpath = '{}_{:03d}w_{:03d}s.txt'.format(
            self.hparams.output_file_root,
            self.hparams.n_ways,
            self.hparams.n_shots)

        print(out_str)
        
        with open(out_fpath, 'w') as f:
            f.write(out_str)

        return {}
    
    def configure_optimizers(self): 
               
        if self.hparams.architecture == 'conv4':
            optimizer = torch.optim.Adam(
                [{
                    'params': self.encoder.parameters(), 
                    'lr':self.hparams.learning_rate
                 },
                 {
                     'params': self.feat.parameters(), 
                     'lr': self.hparams.learning_rate \
                             * self.hparams.lr_multiplier
                 }])  
        else:
            optimizer = torch.optim.SGD(
                [{
                    'params': self.encoder.parameters(), 
                    'lr':self.hparams.learning_rate
                 },
                 {
                     'params': self.feat.parameters(), 
                     'lr': self.hparams.learning_rate \
                             * self.hparams.lr_multiplier
                 }],
                momentum=0.9,
                nesterov=True,
                weight_decay=self.hparams.weight_decay
            )
    
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                self.hparams.step, 
                gamma=self.hparams.lr_gamma)          

        return {'optimizer':optimizer, 'lr_scheduler':lr_scheduler}

    def prepare_data(self):

        train, val, test = get_dataset(self.hparams.dataset, 
                                      self.hparams.modality,
                                      self.hparams.use_aug,
                                      self.extra_args)

        self.train_dataset = train
        self.val_dataset = val
        self.test_dataset = test

    def _build_dataloader(self, split):

        if split == 'val':

            val_sampler_params = [self.val_dataset.labels,
                          self.hparams.n_ls_val_iters,
                          self.hparams.n_ls_val_ways,
                          self.hparams.n_ls_val_shots,
                          self.hparams.n_ls_val_queries]

            val_sampler = CategoriesSampler(*val_sampler_params)
            
            loader1 = DataLoader(
                        self.val_dataset,
                        batch_sampler=val_sampler,
                        shuffle = False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True,
                        drop_last = False)
            
            train_sampler_params = [self.train_dataset.labels,
                      self.hparams.n_ls_val_iters,
                      self.hparams.n_ls_val_ways,
                      self.hparams.n_ls_val_shots,
                      self.hparams.n_ls_val_queries]

            train_sampler = CategoriesSampler(*train_sampler_params)

            loader2 = DataLoader(
                        self.train_dataset,
                        batch_sampler=train_sampler,
                        shuffle = False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True,
                        drop_last = False)

            return [loader1, loader2]
        
        elif split == 'train': 
                
            sampler_params = [self.train_dataset.labels,
                      self.hparams.n_ls_train_iters,
                      self.hparams.n_ls_train_ways,
                      self.hparams.n_ls_train_shots,
                      self.hparams.n_ls_train_queries]

            train_sampler = CategoriesSampler(*sampler_params)

            loader = DataLoader(
                            self.train_dataset,
                            batch_sampler=train_sampler,
                            shuffle = False,
                            num_workers=self.hparams.num_workers,
                            pin_memory=True,
                            drop_last=False)

            return loader

        elif split == 'test':

            sampler_params = [self.test_dataset.labels,
                          self.hparams.n_ls_test_iters,
                          self.hparams.n_ways,
                          self.hparams.n_shots,
                          self.hparams.n_ls_test_queries]
            
            sampler = CategoriesSampler(*sampler_params)

            loader = DataLoader(
                        self.test_dataset,
                        batch_sampler=sampler,
                        shuffle = False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True,
                        drop_last = False)

            return loader


    def train_dataloader(self):
        loader = self._build_dataloader(split='train')
        return loader

    def val_dataloader(self):
        loader = self._build_dataloader(split='val')
        return loader

    def test_dataloader(self):
        loaders = self._build_dataloader(split='test')
        return loaders

