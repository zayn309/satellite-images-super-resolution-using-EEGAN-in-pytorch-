
import os
import shutil
import pandas as pd

LR_source = 'raw_data\Dataset'
HR_source = 'raw_data\Dataset'

LR_dis = 'data\SR_data'
HR_dis = 'data\SR_data'

df = pd.read_csv('data\meta_data.csv')

"""
the flag being true means that the image is not in right shape or most of the pixels is just zeros
the csv file is made in a separated notebook

"""

for index, row in df.iterrows():
    if not row['flag']:
        lr_path_source = os.path.join(LR_source, row['LR_path'])
        hr_path_source = os.path.join(HR_source, row['HR_path'])
        lr_path_dest = os.path.join(LR_dis, row['LR_path'])
        hr_path_dest = os.path.join(HR_dis, row['HR_path'])
        
        shutil.move(lr_path_source, lr_path_dest)
        
        shutil.move(hr_path_source, hr_path_dest)

