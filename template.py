import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "sr"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/{project_name}/base/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/model/__init__.py",
    f"src/{project_name}/data_loader/__init__.py",
    f"src/{project_name}/test/__init__.py",
    f"src/{project_name}/trainer/__init__.py",
    f"src/{project_name}/datsets/__init__.py",
    "__init__.py",
    "config.json",
    "config_GAN.json",
    "requirements.txt",
    "setup.py",
    "saved/"
    
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")
