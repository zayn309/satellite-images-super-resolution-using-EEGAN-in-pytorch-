from sr.data_loader.data_loaders import SR_dataLoader
import tifffile as tiff
from sr.utils.utils import (get_config, plot_examples)
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt 
# print(mp.cpu_count())

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
