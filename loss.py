from tinygrad.tensor import Tensor

from vgg16 import VGG16
from resnet import ResNet

def gram(a): # (bs, channels, width, height)
    a = a.permute(0,2,3,1) # (bs, w, h, c)
    a = (a+1).sqrt()-1
    gram = Tensor.einsum('bhwc, bhwd -> bcd', a, a)
    return gram / (a.shape[-2] * a.shape[-3])

def gram_loss(x, y):
    return (gram(x) - gram(y)).square().mean(axis=(-1,-2))

def style_loss_fn(texture): # (1,ch,w,h)
    layers = [1, 6, 11, 18, 25]
    features = []
    vgg16 = VGG16()
    y = list(vgg16.get_features(texture, layers))
    def loss(x):
        x = x[:,:3,:,:]
        x = vgg16.get_features(x, layers)
        return Tensor.stack([gram_loss(x, y) for x,y in zip(x,y)]).sum()
    return loss

def resnet_loss_fn(texture):
    model = ResNet(18)
    model.load_from_pretrained()
    def resnetloss(x):
        x = x[:,:3,:,:]
        clipLoss = x.clip(0,1).sub(x).abs().mean()
        x = model(x)
        return -x[3][3][4].mean() + clipLoss*1e-4
    return resnetloss

loss_fn = resnet_loss_fn
