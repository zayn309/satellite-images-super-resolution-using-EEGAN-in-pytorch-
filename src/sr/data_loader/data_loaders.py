from torchvision import datasets, utils
from sr.base.base_data_loader import BaseDataLoader
from sr.datsets.sr_dataset import SrDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,
    BboxParams, RandomCrop, Normalize, Resize, VerticalFlip
)

from sr.utils.utils import get_config

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0, training=True):
        trsfm = A.Compose([
            A.Normalize((0.1307,), (0.3081,)),
            A.ToTensor(),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class SR_dataLoader(BaseDataLoader):
    
    def __init__(self,config):
        
        self.config = config
        
        self.batch_size = self.config.data_loader.args.batch_size
        self.data_dir_HR = self.config.data_loader.args.data_dir_HR
        self.data_dir_LR = self.config.data_loader.args.data_dir_LR
        self.shuffle = self.config.data_loader.args.shuffle
        self.validation_split = self.config.data_loader.args.validation_split
        self.num_workers = self.config.data_loader.args.num_workers
        
        lr_data_transforms = A.Compose([
            A.Resize(64,64),
            ToTensorV2(),
        ])
        hr_data_transforms = A.Compose([
            A.Resize(256,256),
            ToTensorV2(),
        ])
        both_transorms = A.Compose([
            A.HorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
            A.RandomRotate90(p=0.5),
        ],additional_targets={'target':"image"})
        self.dataset = SrDataset(LR_root=self.data_dir_LR,HR_root=self.data_dir_HR,
                                 lr_transform = lr_data_transforms, hr_transform = hr_data_transforms, both_transorms = both_transorms )
        super().__init__(self.dataset, self.batch_size, self.shuffle, self.validation_split, self.num_workers)
        
def test():
    sr_loader = SR_dataLoader(config= get_config('config.json'))
    data_loader_val = sr_loader.split_validation()
    print(f'the number of training batches: {len(sr_loader)}')
    print(f'the number of validation batches: {len(data_loader_val)}')
        