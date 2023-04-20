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
x_train, y_train = loadermodule.dataloader(config['data']['train_dir'])
x_train, y_train = preprocess(x_train, y_train, config)

def run_experiment(model):
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

    ckpt_name = config['train']['ckpt_name']
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        ckpt_name + '.h5',
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=config['train']['batch_size'],
        epochs=config['train']['epochs'],
        validation_split=0.1,
        callbacks=[checkpoint_callback]
    )

    # model.load_weights(checkpoint_filepath)
    # _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history

loadermodule = module_loader('model_main', config['model']['model_loader'])
model = loadermodule.model_main()
history = run_experiment(model)