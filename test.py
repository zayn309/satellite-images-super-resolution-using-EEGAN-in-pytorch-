# from sr.data_loader.data_loaders import SR_dataLoader
# import tifffile as tiff
# from sr.utils.utils import (get_config, plot_examples)
# import os
# from pathlib import Path
# import torch
# import numpy as np
# import matplotlib.pyplot as plt 
# from tqdm import tqdm 
# print(mp.cpu_count())
from sr.utils.utils import dict2str
# class ExampleModel(nn.Module):
#     def __init__(self):
#         super(ExampleModel, self).__init__()
#         self.fc = nn.Linear(10, 10)
#     def __str__(self) -> str:
#         return self.__class__.__name__
#     def forward(self, x):
#         return self.fc(x)
    
# model = ExampleModel()
# opt_G = Adam(model.parameters(), lr=0.001)
# opt_D = Adam(model.parameters(), lr=0.001)
# config = get_config('config.json')

# trainer = BaseTrainer(model=ExampleModel, opt_G=opt_G,opt_D=opt_D, config=config,logger=logger)

# # trainer.save_checkpoint(1)
# #trainer.resume_checkpoint("saved\check points\checkpoint-epoch1.pth")

# trainer.train()
# config = get_config("config.json")
# data_loader = SR_dataLoader(config=config)
# _, hr = 

# plot_examples()

result = {
    "results of epoch ": 1,
    'psnr': 30.55,
    'ssim' : 0.6565 ,
    'mse': 2.46452,
    'vgg Loss': 502.012}
print(dict2str(result))
