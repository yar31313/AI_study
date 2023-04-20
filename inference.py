import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import os

from utils.preprocess import preprocess
from utils.util import *

import json
with open('./config/ViT_20230413.json', 'r') as f: config = json.load(f)

loadermodule = module_loader('dataloader', config['data']['data_loader'])
x_test, y_test = loadermodule.dataloader(config['data']['test_dir'])
x_test, y_test = preprocess(x_test, y_test, config)

def run_eval(model):

    optimizer = tfa.optimizers.AdamW(
        learning_rate=config['train']['learning_rate'], weight_decay=config['train']['decay_rate']
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )


    model.load_weights(config['train']['ckpt_name']+'.h5')
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return 

loadermodule = module_loader('model_main', config['model']['model_loader'])
model = loadermodule.model_main(config)
run_eval(model)