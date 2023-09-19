import os 
import sys 
import logging

logging_string = '''[Time: %(asctime)s]
[logging level: %(levelname)s]
[message: %(message)s]
-----------------------------------'''
log_dir = 'logs'
log_filepath = os.path.join(log_dir,"running_log.log")
os.makedirs(log_dir,exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=logging_string,
    
    handlers= [
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SR")
