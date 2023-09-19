import json
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import os
from box import ConfigBox
import random
import numpy as np
import tifffile as tiff
import torch

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


def plot_examples(I_Base, lap, learned_lap, I_sr, config):
    def save_images(image_tensor, save_dir, prefix):
        for i, image in enumerate(image_tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            save_path = str(Path(save_dir) / f"{prefix}_{counter}.tiff" )
            tiff.imsave(save_path, image)
    
    if not isinstance(I_Base, torch.Tensor):
        raise ValueError("Input batch should be a torch.Tensor")

    if len(I_Base.shape) != 4:
        raise ValueError("Input batch should be in the shape of (batch_size, channels, width, height)")

    counter = len(os.listdir(Path(config.examples_dir.base_image))) + 1
    
    # Save I_Base images
    save_images(I_Base, config.examples_dir.base_image, "I_Base")
    
    # Save lap images
    save_images(lap, config.examples_dir.edge, "lap")
    
    # Save learned_lap images
    save_images(learned_lap, config.examples_dir.learned_edge, "learned_lap")
    
    # Save I_sr images
    save_images(I_sr, config.examples_dir.SR_image, "I_sr")

    # Return the updated counter
    return counter
