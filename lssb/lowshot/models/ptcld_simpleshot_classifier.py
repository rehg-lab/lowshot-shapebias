import torch
from torch.nn import functional as F

import lssb.nets as nets
from lssb.lowshot.models.base_simpleshot_classifier import SimpleShotBase

class PtcldClassifier(SimpleShotBase):

    def __init__(self, hparams):
        super().__init__(hparams)

        self.hparams = hparams
        self.extra_args = {
                'num_points':self.hparams.num_points,
                'use_random_SO3_rotation':self.hparams.use_random_rot}

    def _build_model(self):

        if self.hparams.architecture == 'dgcnn-1':
            feat_level=1
            net = nets.dgcnn(
                num_classes=self.hparams.num_classes,
                feat_level=feat_level
            )

        if self.hparams.architecture == 'dgcnn-2':
            feat_level=2
            net = nets.dgcnn(
                num_classes=self.hparams.num_classes,
                feat_level=feat_level
            )

        if self.hparams.architecture == 'dgcnn-3':
            feat_level=3
            net = nets.dgcnn(
                num_classes=self.hparams.num_classes,
                feat_level=feat_level
            )

        self.net = net
    
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

    def forward(self, x):
        image = x['ptcld']
        output = self.net(image)
        return output

