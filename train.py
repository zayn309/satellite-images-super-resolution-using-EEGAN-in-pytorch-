from sr.utils.utils import get_config
from sr.logger import logger
from sr.trainer.EEGAN_trainer import EEGAN_Trainer

def main(config):
    trainer = EEGAN_Trainer(config,logger)
    trainer.train()
    
    
    
if __name__ == '__main__':
    config = get_config("config.json")
    