import json
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import os
from box import ConfigBox
import random
import tifffile as tiff
import torch
import numpy as np
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
        
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def get_config(config_path):
    content = read_json(config_path)
    config = ConfigBox(content)
    return config 



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def plot_examples(I_Base, lap, learned_lap, I_sr, IHR, ILR, config):
    def save_images(image_tensor, save_dir, prefix, counter):
        for i, image in enumerate(image_tensor):
            image = image.permute(1, 2, 0).cpu().detach().numpy()
            save_path = str(Path(save_dir) / f"{prefix}_{counter}.tiff")
            tiff.imsave(save_path, image)

    if not isinstance(I_Base, torch.Tensor):
        raise ValueError("Input batch should be a torch.Tensor")

    if len(I_Base.shape) != 4:
        raise ValueError("Input batch should be in the shape of (batch_size, channels, width, height)")

    counter = len(os.listdir(Path(config.examples_dir.base_image))) + 1

    # Save I_Base images
    save_images(I_Base, config.examples_dir.base_image, "I_Base", counter)

    # Save lap images
    save_images(lap, config.examples_dir.edge, "lap", counter)

    # Save learned_lap images
    save_images(learned_lap, config.examples_dir.learned_edge, "learned_lap", counter)

    # Save I_sr images
    save_images(I_sr, config.examples_dir.SR_image, "I_sr", counter)

    # Save IHR images
    save_images(IHR, config.examples_dir.IHR, "IHR", counter)

    # Save ILR images
    save_images(ILR, config.examples_dir.ILR, "ILR", counter)

    # Return the updated counter
    return counter



def apply_pca(images,DEVICE):
    """
    Apply PCA to compress the number of channels from 4 to 3.

    Args:
        images (torch.Tensor): Tensor of images with shape (batch_size, channels, width, height).

    Returns:
        torch.Tensor: Compressed images with shape (batch_size, 3, width, height).
    """
    batch_size, channels, width, height = images.shape

    # Convert to numpy array for PCA
    images_np = images.detach().cpu().numpy()

    # Reshape images to (batch_size, channels, -1)
    images_reshaped = np.reshape(images_np, (batch_size, channels, -1))

    # Reshape to (batch_size, -1, channels) for PCA
    images_for_pca = np.transpose(images_reshaped, (0, 2, 1))

    # Reshape back to (batch_size * width * height, channels) for PCA
    images_for_pca = np.reshape(images_for_pca, (-1, channels))

    # Apply PCA
    pca = PCA(n_components=3)
    compressed_images = pca.fit_transform(images_for_pca)

    # Reshape back to (batch_size, width, height, 3)
    compressed_images = np.reshape(compressed_images, (batch_size, width, height, 3))

    # Reshape to (batch_size, 3, width, height)
    compressed_images = np.transpose(compressed_images, (0, 3, 1, 2))

    # Convert back to torch.Tensor
    compressed_images = torch.from_numpy(compressed_images).to(DEVICE).double()

    return compressed_images


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)