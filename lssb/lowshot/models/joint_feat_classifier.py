import torch
import os
import pdb
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from argparse import ArgumentParser

import lssb.nets as nets
from lssb.data.sampler import CategoriesSampler
from torch.nn import functional as F

from lssb.lowshot.models.base_feat_classifier import FeatBase
from lssb.lowshot.models.joint_simpleshot_classifier import JointClassifier as j_simpleshot
#data loading

class JointClassifier(FeatBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.extra_args = {
                "feat_dict_file":self.hparams.feat_dict_file}

        self.hparams = hparams
    
    def _build_model(self):

        print("Loading pretrained encoder from", 
                self.hparams.encoder_path_resnet)
        
        pw_model = j_simpleshot.load_from_checkpoint(self.hparams.encoder_path_resnet)

        self.img_encoder = pw_model.net

        self.feat = nets.feat(
                encoder='resnet18',
                use_euclidean=self.hparams.use_euclidean,
                temp1=self.hparams.temperature1,
                temp2=self.hparams.temperature2,
                shot=self.hparams.n_ls_train_shots,
                way=self.hparams.n_ls_train_ways,
                query=self.hparams.n_ls_train_queries)


    def forward(self, x, n_ways, n_shots, n_queries):
        image = x['image']

        
        img_embed = self.img_encoder(image)['embed']
        pc_embed = x['gt_embed']
        
        #for feat we need to make the queries into 
        # [1,2,3,4,1,2,3,4] rather than
        # [1,1,1,1,2,2,2,2,3,3,3,3] etc
        
        if self.hparams.use_pc_for_lowshot:
            shots = (img_embed[0:n_shots * n_ways] + pc_embed[0:n_shots*n_ways])/2
        else:
            shots = img_embed[0:n_shots * n_ways]

        queries = img_embed[n_shots*n_ways:]

        q_idx = torch.arange(n_queries).repeat_interleave(n_ways) \
                + torch.arange(0,
                        n_queries*n_ways, 
                        n_queries).repeat(n_queries)
        s_idx = torch.arange(n_shots).repeat_interleave(n_ways) \
                + torch.arange(0,
                        n_shots*n_ways, 
                        n_shots).repeat(n_shots)
        
        shots=shots[s_idx]
        queries=queries[q_idx] 

        support_idx, query_idx = self.split_instances(n_ways, n_shots, n_queries) 
        instance_emb = torch.cat([shots, queries])
        
        if self.training:
            logits, reg_logits = self.feat(
                instance_emb,
                support_idx,
                query_idx)

            return logits, reg_logits

        else:
            logits = self.feat(
                instance_emb,
                support_idx,
                query_idx)

            return logits
    
    def configure_optimizers(self): 
        
        optimizer = torch.optim.SGD(
            [{
                'params': self.img_encoder.parameters(), 
                'lr':self.hparams.learning_rate
             },
             {
                 'params': self.feat.parameters(), 
                 'lr': self.hparams.learning_rate \
                         * self.hparams.lr_multiplier
             }],
            #momentum=0.9,
            #nesterov=True,
            weight_decay=self.hparams.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                self.hparams.step, 
                gamma=self.hparams.lr_gamma)          

        return {'optimizer':optimizer, 'lr_scheduler':lr_scheduler}

        return {'optimizer':optimizer}


