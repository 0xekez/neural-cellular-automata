from train import load_model
from tinygrad.tensor import Tensor
import numpy as np
from PIL import Image

def create_gif(gif_path, start=0, end=300, fps=60):
    model = load_model("model.safetensors")
    images = []
    x = Tensor.zeros(1, 12, 224, 224)
    for i in range(end+1):
        x = (x + model(x))
        if i >= start:
            xp = x[0][:3,:,:] # take RGB channels
            xp = xp.permute(1, 2, 0) # (height, width, channels)
            xp = xp.numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            images.append(xp)
    images = [Image.fromarray(np.clip(image*255, 0, 255).astype(np.uint8)) for image in images]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=1000/fps, loop=0)
