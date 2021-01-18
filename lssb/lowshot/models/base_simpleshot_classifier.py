import torch 
import os
import pytorch_lightning as pl
import numpy as np

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, models

from lssb.data.sampler import CategoriesSampler
from lssb.lowshot.utils.train_utils import get_dataset
from lssb.lowshot.utils.simpleshot_utils import run_lowshot_testing, run_lowshot_validation

class SimpleShotBase(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        
        self.hparams = hparams

        # extra args is used to give additional parameters when
        # creating the data loaders for learning with point clouds
        # or learning the joint point cloud/image model

        self.extra_args = None
        self._build_model()
 
    def _build_model(self):
        """
            to be implemented by either "image_simpleshot_classifier"
            or joint_simpleshot_classifier" subclasses
        """

        raise NotImplementedError
   
    def forward(self, x):
        """
            to be implemented by either "image_simpleshot_classifier" 
            or "joint_simpleshot_classifier" subclasses
        """

        raise NotImplementedError

    def training_step(self, batch, batch_nb):
        
        output = self(batch)
        
        labels = batch['labels']
        logits = output['logits']
        
        loss = F.cross_entropy(logits, labels) 

        tensorboard_logs = {'train/batch_loss': loss}

        correct = (torch.argmax(logits, dim=1) == labels).float()

        return {'loss': loss, 
                'log':tensorboard_logs, 
                'correct':correct, 
                'labels':labels}
    
    def validation_step(self, batch, batch_nb, loader_idx):

        output = self(batch)
        
        labels = batch['labels']
        embed = output['embed']
        logits = output['logits']
        
        return {'labels': labels,
                'embed': embed,
                'logits': logits}

    def test_step(self, batch, batch_idx, loader_idx):
        
        output = self(batch)
        
        test_labels = batch['labels']
        test_embed = output['embed']
        
        return {
            'embed':test_embed,
            'labels':test_labels,
            }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        correct = torch.cat([x['correct'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        avg_inst_acc = correct.mean()
        
        per_class_accs = [correct[labels==label].mean() \
                          for label in set(self.train_dataset.labels)]
        
        avg_class_acc = torch.Tensor(per_class_accs).mean()
        
        tensorboard_logs = {'train/loss_epoch': avg_loss,
                            'train/inst_acc': avg_inst_acc, 
                            'train/class_acc': avg_class_acc}
        
        return {'train_loss': avg_loss, 'log': tensorboard_logs}
   
    def validation_epoch_end(self, outputs):
        
        ## validation embeddings and validatin labels
        embeds = torch.stack([x['embed'] for x in outputs[0]])
        labels = torch.stack([x['labels'] for x in outputs[0]])

        ## computing training classification accuracy ##
        train_logits = torch.cat([x['logits'] for x in outputs[1]])
        train_labels = torch.cat([x['labels'] for x in outputs[1]])
        
        correct = (torch.argmax(train_logits, dim=1) == train_labels).float()
        per_class_accs = [correct[train_labels==label].mean() \
                          for label in set(self.train_dataset.labels)]
        avg_class_acc = torch.Tensor(per_class_accs).mean()
        
        ## computing validation lowshot accuracy ##
        val_ls_acc = run_lowshot_validation(
                embeds, 
                labels, 
                self.hparams.n_ls_val_shots,
                self.hparams.n_ls_val_ways,
                self.hparams.val_metric
            )

        tensorboard_logs = {'val/lowshot_acc':val_ls_acc,
                            'val/train_acc':avg_class_acc}
        return {'val_acc': val_ls_acc, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        train_embeds= torch.cat([x['embed'].cpu()  for x in outputs[0]])
        mean_embed = torch.mean(train_embeds, axis=0).numpy()

        test_embeds = torch.stack([x['embed'].cpu() for x in outputs[1]])
        test_labels  = torch.stack([x['labels'].cpu() for x in outputs[1]])
        
        test_embeds = test_embeds.numpy()
        test_labels = test_labels.numpy()

        out_str, out_dict =  run_lowshot_testing(
                mean_embed, 
                test_embeds, 
                test_labels, 
                self.hparams.n_ls_test_iters,
                self.hparams.n_shots,
                self.hparams.n_ways,
                self.hparams.n_ls_test_queries)
        
        np_out_fpath = '{}_{:03d}w_{:03d}s.npz'.format(
            self.hparams.output_file_root,
            self.hparams.n_ways,
            self.hparams.n_shots)
       
        out_fpath = '{}_{:03d}w_{:03d}s.txt'.format(
            self.hparams.output_file_root,
            self.hparams.n_ways,
            self.hparams.n_shots)
        
        with open(out_fpath, 'w') as f:
            f.write(out_str)
        
        np.savez(np_out_fpath, arr=out_dict)

        return {}

    def configure_optimizers(self): 

        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(),
                                   lr=self.hparams.learning_rate,
                                   weight_decay=self.hparams.weight_decay)
        
        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.net.parameters(),
                                   lr=self.hparams.learning_rate,
                                   momentum=0.9,
                                   weight_decay=self.hparams.weight_decay)
        
        milestones = [int(.7 * self.hparams.max_epochs), int(.9 * self.hparams.max_epochs)]

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                         milestones=milestones,
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

            sampler_params = [self.val_dataset.labels,
                          self.hparams.n_ls_val_iters,
                          self.hparams.n_ls_val_ways,
                          self.hparams.n_ls_val_shots,
                          self.hparams.n_ls_val_queries]

            sampler = CategoriesSampler(*sampler_params)
            
            loader1 = DataLoader(
                        self.val_dataset,
                        batch_sampler=sampler,
                        shuffle = False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True)
            
            loader2 = DataLoader(
                        self.train_dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle = False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True,
                        drop_last=True)
            
            return [loader1, loader2]
        
        elif split == 'train': 
            
            if self.hparams.use_train_sampler:
                smplr = torch.utils.data.RandomSampler(
                        self.train_dataset,
                        replacement=True,
                        num_samples=20000)

                loader = DataLoader(
                            self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            sampler=smplr,
                            num_workers=self.hparams.num_workers,
                            pin_memory=True,
                            drop_last = True)
                
                return loader

            else:
                loader = DataLoader(
                            self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            shuffle = True,
                            num_workers=self.hparams.num_workers,
                            pin_memory=True,
                            drop_last = True)
                       
                return loader

        elif split == 'test':
            
            sampler_params = [self.test_dataset.labels,
                              self.hparams.n_ls_test_iters,
                              self.hparams.n_ways,
                              self.hparams.n_shots,
                              self.hparams.n_ls_test_queries]
                
            sampler = CategoriesSampler(*sampler_params)

            loader1 = DataLoader(
                        self.train_dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle = False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True,
                        drop_last = False)
            
            loader2 = DataLoader(
                        self.test_dataset,
                        batch_sampler=sampler,
                        shuffle = False,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True,
                        drop_last = False)

            return [loader1, loader2]


    def train_dataloader(self):
        loader = self._build_dataloader(split='train')
        return loader

    def val_dataloader(self):
        loader = self._build_dataloader(split='val')
        return loader

    def test_dataloader(self):
        loaders = self._build_dataloader(split='test')
        return loaders
    
