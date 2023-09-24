from torch.utils.data import Dataset
import glob
import torch
import tifffile as tiff
import numpy as np

class SrDataset(Dataset):
    def __init__(self,LR_root , HR_root,lr_transform = None,hr_transform = None ,both_transorms = None):
        
        self.LR_root = LR_root
        self.HR_root = HR_root
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform
        self.both_transorms = both_transorms
        self.LR_images = list(sorted(set(glob.glob(self.LR_root+"*.tif"))))
        self.HR_images = list(sorted(set(glob.glob(self.HR_root+"*.tif"))))
        
        
    def __len__(self):
        return len(self.HR_images)
    
    def _scaleCCC(self,x):
        x = np.where(np.isnan(x), 0, x)
        return((x - np.nanpercentile(x, 2))/(np.nanpercentile(x, 98) - np.nanpercentile(x,2)))
    
    def __getitem__(self, index):
        lr_image_path = self.LR_images[index]
        hr_image_path = self.HR_images[index]

        lr_image = self._scaleCCC(np.array(tiff.imread(lr_image_path), dtype='float32'))
        hr_image = self._scaleCCC(np.array(tiff.imread(hr_image_path), dtype='float32'))
        if self.both_transorms:
            transformed = self.both_transorms(image = lr_image,target = hr_image)
            lr_image = transformed["image"]
            hr_image = transformed["target"]
        if self.lr_transform:
            lr_image = self.lr_transform(image =lr_image)['image']
        if self.hr_transform:
            hr_image = self.hr_transform(image = hr_image)['image']
            
        return lr_image, hr_image
