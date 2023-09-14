from torchvision import datasets, utils
from sr.base.base_data_loader import BaseDataLoader
from sr.datsets.sr_dataset import SrDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
    
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        lr_data_transforms = A.Compose([
            A.Resize(64,64),
            A.Normalize( 
                mean=[0.36216428, 0.34984875, 0.31362289, 0.64904877],
                std=[0.03167508, 0.0365381,  0.05690316, 0.10670577]
            ),
            ToTensorV2(),
        ])
        hr_data_transforms = A.Compose([
            A.Resize(256,256),
            A.Normalize( 
                mean=[0.08677412, 0.12857, 0.10717302, 0.63853952],
                std=[0.02621013, 0.03286405, 0.05179945, 0.1277203]
            ),
            ToTensorV2(),
        ])
        self.dataset = SrDataset(lr_transform = lr_data_transforms, hr_transform = hr_data_transforms )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
def test():
    sr_loader = SR_dataLoader(batch_size=32, shuffle=True, validation_split=0.2, num_workers=0)
    for batch in sr_loader:
        lr,hr = batch
        print(lr.shape)
        print(hr.shape)
        break
test()