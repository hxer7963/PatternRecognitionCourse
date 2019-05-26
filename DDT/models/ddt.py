import os

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDT(object):

    def __init__(self, features):    
        super(DDT, self).__init__()
        # K is the # of deep self.descriptors which is a Channel-dim vectors.
        self.features = features

    def pca(self):
        N, C, H, W = self.features.shape
        K = N * H * W
        x_mean = (self.features.sum(dim=(3, 2, 0)) / K).view(1, -1, 1, 1)
        assert x_mean.shape == (1, C, 1, 1)
        features = self.features - x_mean
        # self.descriptors.shape = (C, N * H * W)
        self.descriptors = features.view(N, C, -1).permute(1, 0, 2).contiguous().view(C, -1)
        assert self.descriptors.shape == (C, N*H*W)
        # C by C covariance matrix
        cov = torch.matmul(self.descriptors, self.descriptors.t()) / K
        _, eigvec = torch.eig(cov, eigenvectors=True)

        self.main_projection = eigvec[:, 0]
        # indictor matrix $P = \Xi^T \times meaned_features

    def project_to_indictor_matrices(self):
        N, C, H, W = self.features.shape
        self.indictor_matrices = torch.matmul(self.main_projection, self.descriptors).view(N, H, W)

        maxv = self.indictor_matrices.max()
        minv = self.indictor_matrices.min()
        self.indictor_matrices *= (maxv + minv) / torch.abs(maxv + minv)
        
        # filter negative correlation
        self.indictor_matrices = torch.clamp(self.indictor_matrices, min=0)
        # normalize each indictor matrix(image)
        maxv = self.indictor_matrices.view(N, -1).max(dim=1)[0].view(-1, 1, 1)
        self.indictor_matrices /= maxv
        """ resize to the original image size by nearest interpolate 
        Note: bilinear is only support for 4-D tensor. """
        self.indictor_matrices = F.interpolate(self.indictor_matrices.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False) * 255.
        # torch.save(self.indictor_matrices, pca_model_pth)

    def visualize(self, category, images_pth):
        masked_img = []
        for idx, pth in enumerate(images_pth):          
            img = cv2.resize(cv2.imread(pth), (224, 224))
            same_image = [img]
            self.bbox_img = img.copy()
            # mask
            mask = self.indictor_matrices[idx].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy()
            mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
            new_img = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)
            masked_img.append(new_img)
            same_image.append(new_img)
            try:
                from skimage.measure import regionprops
                indictor = self.indictor_matrices[idx].squeeze().cpu().numpy()
                self.bin_img = np.zeros_like(indictor, dtype=np.uint8)
                self.bin_img[indictor > 0] = 1
                regions = regionprops(self.bin_img)
                max_cc = max(regions, key=lambda region: region.area)
                min_r, min_c, max_r, max_c = max_cc.bbox
                # min_r, min_c, max_r, max_c = self.regionprops()
                cv2.rectangle(self.bbox_img, (min_c, min_r+30), (max_c, max_r), (0, 255, 0), 2)
                same_image.append(self.bbox_img)                
            except ValueError as e:
                print(e)
            same_image = np.concatenate(same_image, 1)
            img_pth = './data/{}/ddt/{}'.format(category, pth.split('/')[-1])
            cv2.imwrite(img_pth, same_image)
        masked_image = np.concatenate(masked_img, 1)
        cv2.imwrite('./data/{}.jpg'.format(category), masked_image)