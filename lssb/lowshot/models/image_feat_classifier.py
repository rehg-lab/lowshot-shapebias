import torch
import os
import pdb
import pytorch_lightning as pl

import lssb.nets as nets
from torch.nn import functional as F

from lssb.lowshot.models.base_feat_classifier import FeatBase
from lssb.lowshot.models.image_simpleshot_classifier import ImageClassifier as simpleshot

class ImageClassifier(FeatBase):

    def __init__(self, hparams):
        super().__init__(hparams)

        self.hparams = hparams
    
    def _build_model(self):

        print("Loading pretrained encoder from", 
                self.hparams.encoder_path)
        pt_model = simpleshot.load_from_checkpoint(
                self.hparams.encoder_path)

        self.encoder = pt_model.net

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
        
        embed = self.encoder(image)['embed']
        
        #for feat we need to make the queries into 
        # [1,2,3,4,1,2,3,4] rather than
        # [1,1,1,1,2,2,2,2,3,3,3,3] etc which is what our loader gives us
        
        shots = embed[:n_shots * n_ways]
        queries = embed[n_shots * n_ways:]

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
       

