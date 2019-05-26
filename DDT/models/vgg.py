from pprint import pprint

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms


class VGG19(nn.Module):

    def __init__(self):
        super(VGG19, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.vgg19_relu5_4 = nn.Sequential(
            *list(vgg19.features.children())[:-1]
        )
        """ it's not necessary to set grad. para. to False
        if just perform forward process. """
        for param in self.vgg19_relu5_4.parameters():
            param.requires_grad = False
        # print(self.features)
        
    def forward(self, img):
        x = self.vgg19_relu5_4(img)
        return x
