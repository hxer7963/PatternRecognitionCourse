import os
from PIL import Image
import torch
import torchvision.transforms as tvt
import torchvision.utils as tvu
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import numpy as np

from models import VGG19, DDT
from config import opt
from utils import ImageProcessing

def train(**kwargs):
    opt.parse(kwargs)
    # configure model
    model = VGG19()
    if opt.cuda:
        model.cuda()
    # load data
    train_data = ImageProcessing(data_root=opt.data_root)
    data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    for category in train_data.categories:
        features, img_pth = [], []
        train_data.load_images_path(category)
        for data, pth in data_loader:
            if opt.cuda:
                data = data.cuda()
            img_pth.extend(pth)
            features.append(model.forward(data))
    
        features = torch.cat(features)
        ddt = DDT(features)
        ddt.pca()
        ddt.project_to_indictor_matrices()
        ddt.visualize(category, img_pth)

if __name__ == "__main__":
    import fire
    fire.Fire()