from sr.utils.utils import get_config
from sr.logger import logger
from sr.trainer.EEGAN_trainer import EEGAN_Trainer
import sys

def main(config):
        
    sys.path.append("/kaggle/working/satellite-images-super-resolution-using-EEGAN-in-pytorch-")
    sys.path.append("/kaggle/working/satellite-images-super-resolution-using-EEGAN-in-pytorch-/src")
    sys.path.append("/kaggle/working/satellite-images-super-resolution-using-EEGAN-in-pytorch-/src/sr")
    trainer = EEGAN_Trainer(config,logger)
    trainer.train()
    
    
if __name__ == '__main__':
    config = get_config("config.json")
