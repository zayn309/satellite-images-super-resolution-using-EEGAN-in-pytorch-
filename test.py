# from sr.data_loader.data_loaders import SR_dataLoader
# import tifffile as tiff
from sr.utils.utils import (get_config, plot_examples)
# import os
# from pathlib import Path
# import torch
# import numpy as np
# import matplotlib.pyplot as plt 
# from tqdm import tqdm 
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


import matplotlib.pyplot as plt 
import tifffile as tiff 
import numpy as np


# hr = np.array(torch.permute(target,dims = (0,2,3,1))[0,:,:,:])

# plt.imshow(hr)
# plt.show()
# import numpy as np
# from sklearn.decomposition import PCA

# def apply_pca(images):
#     """
#     Apply PCA to compress the number of channels from 4 to 3.

#     Args:
#         images (numpy.ndarray): Tensor of images with shape (batch_size, channels, width, height).

#     Returns:
#         numpy.ndarray: Compressed images with shape (batch_size, 3, width, height).
#     """
#     batch_size, channels, width, height = images.shape

#     # Reshape images to (batch_size, channels, -1)
#     images_reshaped = np.reshape(images, (batch_size, channels, -1))

#     # Reshape to (batch_size, -1, channels) for PCA
#     images_for_pca = np.transpose(images_reshaped, (0, 2, 1))

#     # Reshape back to (batch_size * width * height, channels) for PCA
#     images_for_pca = np.reshape(images_for_pca, (-1, channels))

#     # Apply PCA
#     pca = PCA(n_components=3)
#     compressed_images = pca.fit_transform(images_for_pca)

#     # Reshape back to (batch_size, width, height, 3)
#     compressed_images = np.reshape(compressed_images, (batch_size, width, height, 3))

#     # Reshape to (batch_size, 3, width, height)
#     compressed_images = np.transpose(compressed_images, (0, 3, 1, 2))
    
#     compressed_images = torch.from_numpy(compressed_images)

#     return compressed_images


# import torch
# from sr.data_loader.data_loaders import SR_dataLoader
# config = get_config("config.json")

# data_loader = SR_dataLoader(config)
# data,target = next(iter(data_loader))
# compressed = apply_pca(target)
# # print()
# plt.imshow(compressed[0,:,:,:])
# plt.show()