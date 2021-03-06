import lssb.nets as nets
from lssb.lowshot.models.base_simpleshot_classifier import SimpleShotBase

class ImageClassifier(SimpleShotBase):

    def __init__(self, hparams):
        super().__init__(hparams)

        self.hparams = hparams
    
    def _build_model(self):
        
        if self.hparams.architecture == 'resnet18':
            net = nets.resnet18(
                num_classes=self.hparams.num_classes,
                feat_dim=512,
                mode='simpleshot'
            )

        if self.hparams.architecture == 'conv4':
            net = nets.conv4(
                num_classes=self.hparams.num_classes
            )
        
        self.net = net

    def forward(self, x):
        image = x['image']
        output = self.net(image)
        return output

