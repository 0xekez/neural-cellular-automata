import random

import numpy as np

from tinygrad.tensor import Tensor

class SamplePool:
    def __init__(self, bs, channels, width, height):
        self.pool = [np.zeros((channels, width, height)) for _ in range(bs)]
        self.indicies = None
    def sample(self, ss):
        assert self.indicies==None, "must commit for each sample call"
        indicies = np.random.choice(len(self.pool), size=ss, replace=False)
        batch = Tensor([self.pool[i] for i in indicies])
        self.indicies = indicies
        return batch
    def commit(self, yp):
        for e, i in enumerate(self.indicies):
            self.pool[i] = yp[e].numpy()
        if random.random() < 0.5:
            self.pool[self.indicies[0]] = np.zeros_like(self.pool[self.indicies[0]])
        self.indicies = None
