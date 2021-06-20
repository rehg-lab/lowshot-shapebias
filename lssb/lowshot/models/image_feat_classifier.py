import torch
import os
import pdb
import pytorch_lightning as pl

import lssb.nets as nets
from torch.nn import functional as F

from lssb.lowshot.models.base_feat_classifier import FeatBase
from lssb.lowshot.models.image_simpleshot_classifier import ImageClassifier as simpleshot
from lssb.lowshot.utils.feat_utils import ConvNet

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

        #below is for recreating performance with conv4
        '''
        self.encoder = ConvNet() 
        model_dict = self.encoder.state_dict()
        pretrained_dict = torch.load(self.hparams.encoder_path)['params']
        #if args.backbone_class == 'ConvNet':
        #    pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        self.encoder.load_state_dict(model_dict)
        '''

        self.feat = nets.feat(
                encoder=self.hparams.architecture,
                use_euclidean=self.hparams.use_euclidean,
                temp1=self.hparams.temperature1,
                temp2=self.hparams.temperature2,
                shot=self.hparams.n_ls_train_shots,
                way=self.hparams.n_ls_train_ways,
                query=self.hparams.n_ls_train_queries)


    def forward(self, x, n_ways, n_shots, n_queries):
        image = x['image']
        
        embed = self.encoder(image)['embed']
        #below is for recreating performance with conv4
        #embed = self.encoder(image)#['embed']
        
        #for feat we need to make the queries into 
        # [1,2,3,4,1,2,3,4] rather than
        # [1,1,1,1,2,2,2,2,3,3,3,3] etc which is what our loader gives us
        
        shots = embed[0:n_shots * n_ways]
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
       

