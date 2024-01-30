from loss import loss_fn
from nca import CA
from batch import SamplePool
from PIL import Image
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, safe_save, safe_load, get_state_dict, load_state_dict
import random

def load_img(img):
    img = Image.open(img)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255
    # normalize using the same params as the torchvision model.
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return Tensor([img.transpose()])

def load_model(filename):
    net = CA(12)
    load_state_dict(net, safe_load(filename))
    return net
def save_model(m, filename):
    safe_save(get_state_dict(m), filename)

if __name__ == "__main__":
    Tensor.seed = 1
    bs = 8
    steps = 2000
    channels=12
    nca = CA(channels)
    pool = SamplePool(256,channels,128,128)
    im = load_img("dots.jpg")
    loss = loss_fn(im)
    Tensor.training = True
    brain = [l.weight for l in nca.layers] + [nca.layers[0].bias]
    opt = Adam(brain,lr=0.002)
    for i in range(0,steps):
        if i==1000:
            opt.lr /= 10
        x = pool.sample(bs)
        for _ in range(random.randint(5,7)):
            x = x + nca(x)
        error = loss(x)
        opt.zero_grad()
        error.backward()
        for b in brain:
            b.grad = b.grad / (b.grad.square().sum().sqrt()+1e-12)
        opt.step()
        pool.commit(x)
        print(i, error.item())
        if (i%10==0):
            save_model(nca, "model.safetensors")
