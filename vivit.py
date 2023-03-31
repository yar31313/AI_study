import os
import time
import zipfile
import numpy as np

import random
import warnings as wa
import pandas as pd

from scipy import ndimage
from time import time

import logging



import SimpleITK as sitk
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.callbacks import TensorBoard,ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold, train_test_split

from data_preprocessing  import *
from dataloader import *
from model import *


import tensorflow_addons as tfa

import logging
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

conf = JsonConfigFileManager('./ViViT_config.json')


#Creating and Configuring Logger

Log_Format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(filename = "logfile.log",
                    stream = sys.stdout, 
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.ERROR)

logger = logging.getLogger()

#Testing our Logger

logger.error("Our First Error Message")
# DATA
DATA_DIR = conf.values.data.DATA_DIR
INPUT_SHAPE = conf.values.data.INPUT_SHAPE
DICOM_SPACING = conf.values.data.DICOM_SPACING
NUM_CLASSES = conf.values.data.NUM_CLASSES

BATCH_SIZE = conf.values.model.BATCH_SIZE
PATCH_SIZE = conf.values.model.PATCH_SIZE
VIVIT_MODEL = conf.values.model.VIVIT_MODEL
LAYER_NORM_EPS = conf.values.model.LAYER_NORM_EPS
PROJECTION_DIM = conf.values.model.PROJECTION_DIM
NUM_HEADS = conf.values.model.NUM_HEADS
NUM_LAYERS = conf.values.model.NUM_LAYERS
LABEL_SMOOTHING = conf.values.model.LABEL_SMOOTHING
STOCHASTIC_DEPTH = conf.values.model.STOCHASTIC_DEPTH

EPOCHS = conf.values.train.EPOCHS
LEARNING_RATE = conf.values.train.LEARNING_RATE
DECAY_RATE = conf.values.train.DECAY_RATE
DECAY_STEP = conf.values.train.DECAY_STEP

OUT_DIR = conf.values.out.OUT_DIR
SAVE_FREQ = conf.values.out.SAVE_FREQ

try: os.listdir(OUT_DIR)
except: os.mkdir(OUT_DIR)

train_tabular = pd.read_csv(DATA_DIR + r"train.csv")
test_tabular = pd.read_csv(DATA_DIR + r"test.csv")
train_img_paths = DATA_DIR + r"train"
test_img_paths = DATA_DIR + r"test"

(train_videos, train_labels) = dataloader(train_tabular,train_img_paths,INPUT_SHAPE)
(test_videos, test_labels) = dataloader(test_tabular,test_img_paths,INPUT_SHAPE)

trainloader = prepare_dataloader(train_videos, train_labels, "train",BATCH_SIZE)
#validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
testloader = prepare_dataloader(test_videos, test_labels, "test")

strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.ReductionToOneDevice())

def run_experiment():
    # Initialize model
    with strategy.scope():
        model = create_vivit_classifier(
            tubelet_embedder=TubeletEmbedding(
                embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE, input_shape=INPUT_SHAPE
            ),
            positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
            vivit_model='STA',
            input_shape=INPUT_SHAPE,
            patch_size=PATCH_SIZE,
            transformer_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            embed_dim=PROJECTION_DIM,
            layer_norm_eps=LAYER_NORM_EPS,
            num_classes=NUM_CLASSES,
        )
        
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # LEARNING_RATE, decay_steps=1000, decay_rate=0.96, staircase=True
        # )
        
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        )
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(OUT_DIR+"/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', save_best_only=True, mode='auto')
        # Train the model.
        _ = model.fit(trainloader, epochs=EPOCHS, validation_data=testloader, shuffle=True, callbacks=[checkpoint,earlystopping],) #, checkpoint, earlystopping

        _, accuracy = model.evaluate(testloader)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

        return model


model = run_experiment()
