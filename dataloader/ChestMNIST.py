import cv2
import os
from time import time
import numpy as np


def dataloader(target_path, config):

    x = []
    y = []
    s = time()
    l = len(os.listdir(target_path+ 'NORMAL/'))
    for i, file in enumerate(os.listdir(target_path+ 'NORMAL/')):
        if i%200==1: print(f'Data loaded :\t{i}/{l},\tEstimated time left : {round((time()-s)*(l/i)-time()+s)}s')
        img = cv2.imread(target_path + 'NORMAL/' + file, cv2.IMREAD_GRAYSCALE)
        x.append(img)
        y.append(0)

    s = time()
    l = len(os.listdir(target_path+ 'PNEUMONIA/'))
    for i, file in enumerate(os.listdir(target_path+ 'PNEUMONIA/')):
        if i%200==1: print(f'Data loaded : {i}/{l},\tEstimated time left : {round((time()-s)*(l/i)-time()+s)}s')
        img = cv2.imread(target_path + 'PNEUMONIA/' + file, cv2.IMREAD_GRAYSCALE)
        x.append(img)
        y.append(1)

    return np.array(x), np.array(y)