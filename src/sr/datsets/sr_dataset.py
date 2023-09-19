from torch.utils.data import Dataset
import glob
import torch
import tifffile as tiff
import numpy as np

class SrDataset(Dataset):
    def __init__(self,LR_root ,HR_root,lr_transform = None,hr_transform = None):
        
        self.LR_root = LR_root
        self.HR_root = HR_root
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.LR_images = list(sorted(set(glob.glob(self.LR_root+"*.tif"))))
        self.HR_images = list(sorted(set(glob.glob(self.HR_root+"*.tif"))))
        
    def __len__(self):
        return len(self.HR_images)
    
    def __getitem__(self, index):
        lr_image_path = self.LR_images[index]
        hr_image_path = self.HR_images[index]

        lr_image = np.array(tiff.imread(lr_image_path), dtype='float32')
        hr_image = np.array(tiff.imread(hr_image_path), dtype='float32')

        if self.lr_transform:
            lr_image = self.lr_transform(image =lr_image)['image']
        if self.hr_transform:
            hr_image = self.hr_transform(image = hr_image)['image']
            

        return lr_image, hr_image
