import cv2
import os
from time import time
import numpy as np
from utils import readTotalVolume, resample_Total, extract_slices

def dataloader(target_path, config):
    x = []
    y = []
    s = time()
    l = len(os.listdir(target_path+ 'benign/'))
    for i, file in enumerate(os.listdir(target_path+ 'benign/')):
        if i%10==1: print(f'Data loaded :\t{i}/{l},\tEstimated time left : {round((time()-s)*(l/i)-time()+s)}s')
        dcm = readTotalVolume(target_path+'benign/'+file)
        dcm = resample_Total(dcm, out_spacing=config['data']['voxel_spacing'])
        dcm = extract_slices(dcm)
        x.append(dcm)
        y.append(0)

    s = time()
    l = len(os.listdir(target_path+ 'malignant/'))
    for i, file in enumerate(os.listdir(target_path+ 'malignant/')):
        if i%10==1: print(f'Data loaded :\t{i}/{l},\tEstimated time left : {round((time()-s)*(l/i)-time()+s)}s')
        dcm = readTotalVolume(target_path+'malignant/'+file)
        dcm = resample_Total(dcm, out_spacing=config['data']['voxel_spacing'])
        dcm = extract_slices(dcm)
        x.append(dcm)
        y.append(0)

    return np.array(x), np.array(y)