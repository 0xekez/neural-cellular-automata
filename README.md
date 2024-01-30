This is an implementation of [_Self-Organising
Textures_](https://distill.pub/selforg/2021/textures/) by Eyvind
Niklasson et. al. using a neural network framework called
[tinygrad](https://github.com/tinygrad/tinygrad).

![](legif.gif)

Install dependencies:

```
python3 -m pip install Pillow numpy matplotlib tinygrad torchvision
```

To train a model:

```
python3 train.py
```

To create a gif called `agif.gif` of the trained or currently training
model:

```
python3 -i
>>> from viz import create_gif
>>> create_gif("agif.gif")
```

Model hyperparamaters can be adjusted in `train.py`, `vgg16.py`
contains a tinygrad VGG-16 model which loads its weights from the
torchvision pretrained model. `loss.py` contains various loss
functions for minimizing style-loss, or maximizing the activation of a
unit the Resnet image classifier.

Tinygrad is _much_ slower than pytorch for training. See `pytorch.py`
for a pytorch version of this software which has been modified to
produce monochrome textures.
