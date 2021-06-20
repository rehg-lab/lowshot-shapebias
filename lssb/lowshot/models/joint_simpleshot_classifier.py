import torch
import os
import pdb
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torch.nn import functional as F
from argparse import ArgumentParser

import lssb.nets as nets
from lssb.data.sampler import CategoriesSampler

from lssb.lowshot.models.base_simpleshot_classifier import SimpleShotBase
from lssb.lowshot.utils.train_utils import get_dataset
from lssb.lowshot.models.image_simpleshot_classifier import ImageClassifier
from lssb.lowshot.utils.simpleshot_utils import run_lowshot_testing, run_lowshot_validation
#data loading

class JointClassifier(SimpleShotBase):
    
    def __init__(self, hparams):
        super().__init__(hparams)

        self.hparams = hparams
        self.extra_args = {'feat_dict_file':self.hparams.feat_dict_file}

    def _build_model(self):
       
        net = nets.resnet18(
            num_classes=self.hparams.num_classes,
            feat_dim=self.hparams.feat_dim,
            mode='feat_extract')
        
        self.net = net

    def forward(self, x):

        image = x['image']
        output = self.net(image)

        return output


    def training_step(self, batch, batch_nb):
        
        output = self(batch)
        gt_embed = batch['gt_embed']
        embed = output['embed']

        return {'embed':embed,
                'gt_embed':gt_embed}
        
    
    def training_step_end(self, outputs):
        embed = outputs['embed']
        gt_embed = outputs['gt_embed']
        
        # compute MSE loss between embeddings
        mse_loss = F.mse_loss(embed, gt_embed, reduction=self.hparams.loss_reduction) 
        
        # \matcal{L}_2 in the paper
        if self.hparams.add_pairwise_dist:
            
            #collect pairwise distances between shape embeddings
            m1 = torch.cdist(gt_embed, gt_embed)

            #collect pairwise distances between resnet embeddings
            m2 = torch.cdist(embed, embed) 
    
            #get the upper triangular elements of each pairwise distance matrix
            vals1 = m1[torch.triu(torch.ones(self.hparams.batch_size, 
                                             self.hparams.batch_size),
                                             diagonal=1) == 1]
            
            vals2 = m2[torch.triu(torch.ones(self.hparams.batch_size, 
                                             self.hparams.batch_size),
                                             diagonal=1) == 1]
            
            # compute MSE loss between the pairiwse distances
            pairwise_dist_term = F.mse_loss(vals1, vals2, reduction=self.hparams.loss_reduction)
            mse_loss += pairwise_dist_term
        
        loss = mse_loss
        tensorboard_logs = {'train/batch_loss': loss}

        return {'loss': loss, 
                'log':tensorboard_logs}
    
    #for multi-gpu training, the batch gets split up between GPUs 
    #so we need to do this first just to get the features and labels for each batch/episode
    def validation_step(self, batch, batch_nb, loader_idx):
        model_output = self(batch)
        output = {}

        labels = batch['labels']
        gt_embed = batch['gt_embed']  
        embed = model_output['embed']
        
        output['labels'] = labels
        output['gt_embed'] = gt_embed
        output['embed'] = embed

        return output
    
    def validation_step_end(self, outputs):
        labels = outputs['labels']
        gt_embed = outputs['gt_embed']
        embed = outputs['embed']
       
        if self.hparams.use_pc_for_lowshot:
            idx = self.hparams.n_ls_val_ways \
                   * self.hparams.n_ls_val_shots
           
            # combining image and shape embedding
            joint_embed = embed.detach().clone() 
            joint_embed[0:idx,:] = (joint_embed[0:idx,:] + gt_embed[0:idx,:])/2

        else:
            joint_embed = embed


        return {'labels': labels,
                'gt_embed':gt_embed,
                'joint_embed':joint_embed,
                'embed': embed}
    
    def test_step(self, batch, batch_idx, loader_idx):
        
        labels = batch['labels']
        output = self(batch)
        
        if loader_idx == 1:
            ### test loader
             
            idx = self.hparams.n_ways \
                   * self.hparams.n_shots

            img_embed = output['embed']
            shape_embed = batch['gt_embed']
            
            # combining image and shape embedding, just for the shots and 
            # not for the queries

            if self.hparams.use_pc_for_lowshot:
                joint_embed = img_embed.detach().clone()
                joint_embed[0:idx,:] = (joint_embed[0:idx,:] + shape_embed[0:idx,:])/2
            else:
                joint_embed = img_embed


        elif loader_idx == 0:
            ### train loader
            img_embed = output['embed']
            shape_embed = batch['gt_embed']

            if self.hparams.use_pc_for_lowshot:
                joint_embed = (img_embed + shape_embed)/2
            else:
                joint_embed = img_embed

       
        return {
                'embed':joint_embed,
                'labels':labels
              }
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        tensorboard_logs = {'train/loss_epoch': avg_loss,
                            'train/mse_loss_epoch': avg_loss}
        
        return {'train_loss': avg_loss, 'log': tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        
        val_embeds = torch.stack([x['embed'] for x in outputs[0]])
        val_joint_embeds = torch.stack([x['joint_embed'] for x in outputs[0]])
        gt_val_embeds = torch.stack([x['gt_embed'] for x in outputs[0]])
        val_labels = torch.stack([x['labels'] for x in outputs[0]])
        
        val_mse = F.mse_loss(val_embeds, gt_val_embeds)

        ## computing training classification accuracy ##
        train_embeds = torch.stack([x['embed'] for x in outputs[1]])
        gt_train_embeds = torch.stack([x['gt_embed'] for x in outputs[1]])

        train_mse = F.mse_loss(train_embeds, gt_train_embeds)
        
        ## computing validation lowshot accuracy ##
        val_ls_acc = run_lowshot_validation(
                val_joint_embeds, 
                val_labels, 
                self.hparams.n_ls_val_shots,
                self.hparams.n_ls_val_ways,
                self.hparams.val_metric
            )


        tensorboard_logs = {'val/lowshot_acc':val_ls_acc,
                            'val/val_mse':val_mse,
                            'val/train_mse':train_mse}

        return {'val_acc': val_ls_acc, 'val_mse': val_mse, 'log': tensorboard_logs}

