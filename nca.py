from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import numpy as np

class Convolution:
    def __init__(self, kernel, channels):
        w, h = kernel.shape
        self.weight = kernel = Tensor(kernel).reshape(1, 1, w, h).repeat((channels, channels, 1, 1))
    def __call__(self, x): # x: (bs, channels, width, height)
        # convolve with wraparound.
        x = Tensor.cat(x[:,:,-1:],x,x[:,:,:1],dim=2)
        x = Tensor.cat(x[:,:,:,-1:],x,x[:,:,:,:1], dim=3)
        return x.conv2d(self.weight, padding=0)

sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
],dtype=np.float32) / 8
sobelY = sobelX.T
laplacian = np.array([
    [1,   2, 1],
    [2, -12, 2],
    [1,   2, 1],
],dtype=np.float32) / 8

hiddenWidth = 96

def stocastic(x, p=0.5):
    b,c,w,h=x.shape
    return x * (Tensor.rand(b,1,w,h) >= p)

class CA:
    def __init__(self, channels):
        self.gradientX = Convolution(sobelX, channels)
        self.gradientY = Convolution(sobelY, channels)
        self.laplacian = Convolution(laplacian, channels)

        self.layers = [
            Linear(4*channels, hiddenWidth, bias=True),
            Linear(hiddenWidth, channels, bias=False)
        ]
        self.layers[-1].weight = Tensor.zeros_like(self.layers[-1].weight)

    def __call__(self, x):
        dx = self.gradientX(x)
        dy = self.gradientY(x)
        l = self.laplacian(x)
        x = Tensor.cat(x, dx, dy, l, dim=1)
        x = x.permute(0,2,3,1) # (bs, ch, w, h) -> (bs, w, h, ch)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = x.relu()
        return stocastic(self.layers[-1](x).permute(0, 3, 1, 2))
