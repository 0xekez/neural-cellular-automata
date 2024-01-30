# VGG-16 model with only the feature layers.

from tinygrad.tensor import Tensor

import torchvision
from tinygrad.nn.state import get_state_dict

class Conv3x3:
    def __init__(self, in_channels, out_channels):
        self.weight = Tensor.uniform(out_channels, in_channels, 3, 3)
        self.bias = Tensor.uniform(out_channels)
    def __call__(self, x):
        return x.conv2d(self.weight).add(self.bias.reshape(1, -1, 1, 1))

class MaxPool2D:
    def __call__(self, x):
        return x.max_pool2d(kernel_size=2, stride=2, dilation=1)

class Relu:
    def __call__(self, x):
        return x.relu()

class VGG16:
    def __init__(self):
        self.features = [
            Conv3x3(3, 64),    # (0)
            Relu(),
            Conv3x3(64, 64),   # (2)
            Relu(),
            MaxPool2D(),

            Conv3x3(64, 128),  # (5)
            Relu(),
            Conv3x3(128, 128), # (7)
            Relu(),
            MaxPool2D(),

            Conv3x3(128, 256), # (10)
            Relu(),
            Conv3x3(256, 256), # (12)
            Relu(),
            Conv3x3(256, 256), # (14)
            Relu(),
            MaxPool2D(),

            Conv3x3(256, 512), # (17)
            Relu(),
            Conv3x3(512, 512), # (19)
            Relu(),
            Conv3x3(512, 512), # (21)
            Relu(),
            MaxPool2D(),

            Conv3x3(512, 512), # (24)
            Relu(),
            Conv3x3(512, 512), # (26)
            Relu(),
            Conv3x3(512, 512), # (28)
            Relu(),
            MaxPool2D(),
        ]
        self.load_from_torchvision()

    def load_from_torchvision(self):
        weights = torchvision.models.vgg16(weights='DEFAULT').state_dict()
        for k,v in get_state_dict(self).items():
            v.assign(weights[k].numpy()).realize()

    def get_features(self, x, layers):
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in layers:
                yield x
