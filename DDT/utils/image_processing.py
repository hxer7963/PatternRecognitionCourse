import os
import glob
from PIL import Image

import torch
from torch import nn
from torch.utils import data
from torchvision import transforms as T
from torch.autograd import Variable

class ImageProcessing(data.Dataset):
    """ encapsulate the methods of load/preprocessing/save """
    def __init__(self, data_root, transforms=None):
        self.data_root = data_root
        self.categories = [d for d in os.listdir(data_root) if os.path.isdir(data_root + d)]
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError('please implement transforms method.')

    def load_images_path(self, sub_dir='', seed=43):
        pics_pth = os.path.join(self.data_root, sub_dir)
        ddt_dir = os.path.join(pics_pth, 'ddt')
        if not os.path.isdir(ddt_dir):
            os.mkdir(ddt_dir)
        print('loading pictures from {}/'.format(pics_pth))
        assert os.path.isdir(pics_pth), "sub_dir is invaild label of this dataset."
        self.images_pth = [os.path.join(pics_pth, filename) for filename in os.listdir(pics_pth) if not filename.startswith('.') and filename != 'ddt']
        # self.images_pth = glob.glob(pics_pth+'/**/*.{}'.format(ext), recursive=True) 

    def __getitem__(self, index):
        assert abs(index) < len(self.images_pth)
        pic_pth = self.images_pth[index]
        """ return digit image & label """
        pth = pic_pth  # dummy variance in this application(DDT)
        # label = None -> will raise classType Error.
        data = Image.open(pic_pth).convert('RGB')
        data = self.transforms(data)
        return data, pth

    def __len__(self):
        return len(self.images_pth)