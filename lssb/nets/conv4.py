from torch import nn

__all__ = ['Conv4']

def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class Conv4(nn.Module):
    def __init__(self, num_classes=None):
        super(Conv4, self).__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)

        if num_classes is not None:
            self.fc = nn.Linear(1600, num_classes)
        else:
            self.fc = None

    def forward(self, x):

        output = {}

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        feature = x.view(x.size(0), -1)

        output['embed'] = feature

        if self.fc is not None:
            logits = self.fc(feature)  
            output['logits'] = logits
        
        return output
