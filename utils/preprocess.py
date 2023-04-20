import albumentations as albu
from skimage import transform

import numpy as np


# import json
# with open('./config/ViT_20230413.json', 'r') as f: config = json.load(f)

def preprocess(x, y, config):
    if 'resize' in config['data']:
        if len(x[0].shape)==2 or x[0].shape[-1] == 1:
            x = np.array([albu.Resize(config['data']['resize'][0],
                                      config['data']['resize'][1])
                                      (image=x_i)['image'] for x_i in x])
        else: x = np.array([transform.resize(x_i, config['data']['resize']) for x_i in x])

    return x, y