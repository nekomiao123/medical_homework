import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFilter
import random


class GaussianBlur:
    """ apply ImageFilter.GaussianBlur to an image """
    def __init__(self, sigma1=0.1, sigma2=2.0):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __call__(self, x):
        return x.filter(ImageFilter.GaussianBlur(random.uniform(self.sigma1, self.sigma2)))

    def __repr__(self):
        return f'GaussianBlur({self.sigma1}, {self.sigma2})'

    def __str__(self):
        return self.__repr__()
