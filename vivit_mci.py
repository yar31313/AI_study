import os
import time
import zipfile
import numpy as np

import random
import warnings as wa
import pandas as pd

from scipy import ndimage
from time import time

import SimpleITK as sitk
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.callbacks import TensorBoard,ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold, train_test_split

from MHA_custom import *

import tensorflow_addons as tfa

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def readTotalVolume(ct_volume_path): #데이터 불러오는 것
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(ct_volume_path)
    reader.SetFileNames(dicom_files)
    retrieved_ct_volume = reader.Execute()
    return retrieved_ct_volume

def resample_Total(input_volume, out_spacing=[1, 1, 1],  is_label=True):
    original_spacing = input_volume.GetSpacing()
    original_size = input_volume.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    #print(out_size)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(input_volume.GetDirection())
    resample.SetOutputOrigin(input_volume.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(input_volume.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(input_volume)

def resize_Total(input_volume, width, height, depth):
    # Resize images to fixed spatial resolution in pixels
    # num_axial_slices = int(input_volume.GetSize()[-1])
    # output_size = [int(input_volume.GetSize()[0]), int(input_volume.GetSize()[1]), num_axial_slices]
    output_size = [width, height, depth]
    scale = np.divide(input_volume.GetSize(), output_size)
    spacing = np.multiply(input_volume.GetSpacing(), scale)
    transform = sitk.AffineTransform(3)
    resized_volume = sitk.Resample(input_volume, output_size, transform, sitk.sitkLinear, input_volume.GetOrigin(),
                                  spacing,input_volume.GetDirection())    
    return resized_volume

def extract_slices(image_volume):
    image_array = sitk.GetArrayFromImage(image_volume)
    image_slices_array = image_array[0::2, :, :]
    return image_slices_array

def normalize(volume):
    """Normalize the volume"""
    min = volume.min()
    max = volume.max()
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

_CHR_IDX = string.ascii_lowercase

def _build_proj_equation(free_dims, bound_dims, output_dims):
    
  """Builds an einsum equation for projections inside multi-head attention."""
  input_str = ""
  kernel_str = ""
  output_str = ""
  bias_axes = ""
  letter_offset = 0
  for i in range(free_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    output_str += char

  letter_offset += free_dims
  for i in range(bound_dims):
    char = _CHR_IDX[i + letter_offset]
    input_str += char
    kernel_str += char

  letter_offset += bound_dims
  for i in range(output_dims):
    char = _CHR_IDX[i + letter_offset]
    kernel_str += char
    output_str += char
    bias_axes += char
  equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

  return equation, bias_axes, len(output_str)

def _get_output_shape(output_rank, known_last_dims):
  return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)

# DATA
DATASET_NAME = "organmnist3d"
BATCH_SIZE = 4
AUTO = tf.data.AUTOTUNE
INPUT_SHAPE = (72,180,180, 1)
NUM_CLASSES = 2

# OPTIMIZER
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6

# TRAINING
EPOCHS = 3000

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

# Model Checkpoint PATH


def dataloader(df,data_path):
  img_concat = []
  label_concat = []
  for i in range(len(df)):
      print("start :",i)
      folder_name = df['patient_ID'].iloc[i]

      patient_path = os.path.join(data_path,folder_name)

      dicom_series = readTotalVolume(patient_path)
      dicom_series = resample_Total(dicom_series)
      dicom_series = resize_Total(dicom_series, 180,180,144)
      dicom_series = extract_slices(dicom_series)
      dicom_series = normalize(dicom_series)
      #img_concat = np.stack((dicom_series, mask), axis=-1)
      
      #dicom_series = np.expand_dims(dicom_series, axis=-1)
      # print(dicom_series.shape)
      # img_concat = np.stack((img_concat,dicom_series), axis=-1)
      img_concat.append(dicom_series)
      ## Tabular
      
      
      ## label
      label_concat.append(df['dx'].iloc[i])
  
  return np.array(img_concat), np.array(label_concat)

train_tabular = pd.read_csv(r"/storage01/users/user/ny_test/CNvsMCI/train.csv")
test_tabular = pd.read_csv(r"/storage01/users/user/ny_test/CNvsMCI/test.csv")


train_img_paths = '/storage01/users/user/ny_test/CNvsMCI/train'
test_img_paths = '/storage01/users/user/ny_test/CNvsMCI/test'

(train_videos, train_labels) = dataloader(train_tabular,train_img_paths)
(test_videos, test_labels) = dataloader(test_tabular,test_img_paths)

@tf.function
def preprocess(frames: tf.Tensor, label: tf.Tensor):
    """Preprocess the frames tensors and parse the labels."""
    # Preprocess images
    frames = tf.image.convert_image_dtype(
        frames[
            ..., tf.newaxis
        ],  # The new axis is to help for further processing with Conv3D layers
        tf.float32,
    )
    # Parse label
    label = tf.cast(label, tf.float32)
    return frames, label


def prepare_dataloader(
    videos: np.ndarray,
    labels: np.ndarray,
    loader_type: str = "train",
    batch_size: int = BATCH_SIZE,
):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(BATCH_SIZE * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader


trainloader = prepare_dataloader(train_videos, train_labels, "train")
# validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
testloader = prepare_dataloader(test_videos, test_labels, "test")

class TubeletEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_size, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))
        # self.flatten_fe = layers.Reshape(target_shape=(int(input_shape[0]/patch_size[0]),
        #                                             int(input_shape[1]/patch_size[1])*int(input_shape[2]/patch_size[2]), embed_dim))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projection': self.projection,
            'flatten': self.flatten,
            #'flatten_fe': self.flatten_fe,
        })
        return config
    
    def call(self, videos, vivit_model):
        projected_patches = self.projection(videos)
        if vivit_model == 'FEE':
            flattened_patches = self.flatten_fe(projected_patches)
        else:
            flattened_patches = self.flatten(projected_patches)        
        return flattened_patches
    
class PositionalEncoder(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config
    
    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = tf.range(start=0, limit=num_tokens, delta=1)

    def call(self, encoded_tokens):
        # Encode the positions and add it to the encoded tokens
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens
    
def create_vivit_classifier(
    tubelet_embedder,
    positional_encoder,
    vivit_model,
    input_shape=INPUT_SHAPE,
    patch_size=PATCH_SIZE,
    transformer_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    embed_dim=PROJECTION_DIM,
    layer_norm_eps=LAYER_NORM_EPS,
    num_classes=NUM_CLASSES,
):
    # Get the input layer
    inputs = layers.Input(shape=input_shape)

    if vivit_model == 'STA':
        # Create patches.
        patches = tubelet_embedder(inputs, vivit_model)
    
        # Encode patches.
        encoded_patches = positional_encoder(patches)

        
        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization and MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
            )(x1, x1)
            print("attention_output :",attention_output.shape)
            # Skip connection
            x2 = layers.Add()([attention_output, encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = layers.Add()([x3, x2])        
        
    elif vivit_model == 'FE':
        # Create patches.
        patches = tubelet_embedder(inputs, vivit_model)
        encoded_patches = positional_encoder(patches) 
        #print("pt_encoded_patches :",encoded_patches.shape)

        for t in range(int(input_shape[0]/patch_size[0])):
            n_t = int(input_shape[0]/patch_size[0])
            n_w = int(input_shape[1]/patch_size[1])
            n_h = int(input_shape[2]/patch_size[2])
            p_encoded_patches = layers.Reshape(target_shape=(n_t, n_w*n_h, embed_dim))(encoded_patches)
            # MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(p_encoded_patches[:,t])
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
            )(x1, x1)

            # Skip connection
            x2 = layers.Add()([attention_output, x1])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)
            
            # Skip connection
            encoded_patch = layers.Add()([x3, x2])
            
            if t == 0: t_encoded_patches = encoded_patch
            else: t_encoded_patches = layers.Concatenate(axis=1)([t_encoded_patches, encoded_patch])
        
        for _ in range(transformer_layers):
            # MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(t_encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
            )(x1, x1)

            # Skip connection
            x2 = layers.Add()([attention_output, t_encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = layers.Add()([x3, x2])

    elif vivit_model == 'FSA':
        # Create patches.
        patches = tubelet_embedder(inputs, vivit_model)
    
        # Encode patches.
        encoded_patches = positional_encoder(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            n_t = int(input_shape[0]/patch_size[0])
            n_w = int(input_shape[1]/patch_size[1])
            n_h = int(input_shape[2]/patch_size[2])
            # Spatial MHSA
            x1 = layers.Reshape(target_shape=(n_t, n_w*n_h, embed_dim))(encoded_patches)
            x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1, attention_axes=(2,3)
            )(x1, x1)
            x2 = layers.Add()([attention_output, x1])

#             x2 = layers.Reshape(target_shape=(n_t, n_w, n_h, embed_dim))(x2)
#             x2 = layers.Permute((2,3,1,4))(x2)

            # Temporal MHSA
#             x2 = layers.Reshape(target_shape=(-1, n_t*embed_dim))(x2)
            x2 = layers.LayerNormalization(epsilon=1e-6)(x2)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1, attention_axes=(1,3)
            )(x2, x2)
            x3 = layers.Add()([attention_output, x2])              
            
#             x3 = layers.Reshape(target_shape=(n_w, n_h, n_t, embed_dim))(x3)
#             x3 = layers.Permute((3,1,2,4))(x3)
            
            x3 = layers.Reshape(target_shape=(-1, embed_dim))(x3)
   
            # Layer Normalization and MLP
            x4 = layers.LayerNormalization(epsilon=1e-6)(x3)
            x4 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x4)

            # Skip connection
            encoded_patches = layers.Add()([x4, x3])
        
    elif vivit_model == 'FDPA':
        # Create patches.
        patches = tubelet_embedder(inputs, vivit_model)
        # Encode patches.
        encoded_patches = positional_encoder(patches)
        
        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            n_t = int(input_shape[0]/patch_size[0])
            n_w = int(input_shape[1]/patch_size[1])
            n_h = int(input_shape[2]/patch_size[2])
            
            x1 = layers.Reshape(target_shape=(n_t, n_w*n_h, embed_dim))(encoded_patches)
            
            # Layer normalization and MHSA
            x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
            sp_attention_output = MultiHeadAttention_nonlinear( #, sp_attention_weights
                num_heads=num_heads//2, key_dim=embed_dim // num_heads, dropout=0.1, attention_axes=(2,3)
            )(x1, x1) #, return_attention_scores=True
            tm_attention_output = MultiHeadAttention_nonlinear( #, tm_attention_weights
                num_heads=num_heads//2, key_dim=embed_dim // num_heads, dropout=0.1, attention_axes=(1,3)
            )(x1, x1) #, return_attention_scores=True
                        
            #print("sp_attention_output :",sp_attention_output.shape)
            #print("sp_attention_weights :",sp_attention_weights.shape)
            #print("tm_attention_output :",tm_attention_output.shape)
            #print("tm_attention_weights :",tm_attention_weights.shape)
                    
            attention_output = layers.Concatenate(axis=-1)([sp_attention_output, tm_attention_output])
            #print("attention_output_1 :",attention_output.shape)
            einsum_equation, bias_axes, output_rank = _build_proj_equation(3, 2, 1)
            
            #print(einsum_equation)
            print(output_rank)
            print(bias_axes)
            
            attention_output = layers.EinsumDense(einsum_equation,
                                                  output_shape=_get_output_shape(output_rank - 1,[embed_dim])
                                                  ,bias_axes=bias_axes)(attention_output)
            
            attention_output = layers.Reshape(target_shape=(-1, embed_dim))(attention_output)            
            # Skip connection
            x2 = layers.Add()([attention_output, encoded_patches])

            # Layer Normalization and MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
                    layers.Dense(units=embed_dim, activation=tf.nn.gelu),
                ]
            )(x3)

            # Skip connection
            encoded_patches = layers.Add()([x3, x2])
    
    else: raise Exception(f'ViViT model method is not defined correctly : \'{vivit_model}\'. Please define a model method as following. [\'STA\', \'FE\', \'FSA\', \'FDPA\']')
        
    # Layer normalization and Global average pooling.
    representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
    representation = layers.GlobalAvgPool1D()(representation)
    
    # Classify outputs.
    outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.ReductionToOneDevice())
        
def run_experiment():
    # Initialize model
    with strategy.scope():
        model = create_vivit_classifier(
            tubelet_embedder=TubeletEmbedding(
                embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE, input_shape=INPUT_SHAPE
            ),
            positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
            vivit_model='STA'
        )
        model.summary()
        #model.load_weights('/storage01/users/user/ny_test/FE_h5/weights.591-0.36.hdf5')
        model.load_weights('/storage01/users/user/ny_test/STA/weights.225-0.26.hdf5')
        #model.load_weights('/storage01/users/user/ny_test/FE_h5/weights.591-0.36.hdf5') # FE
        #model.load_weights('/storage01/users/user/ny_test/FSA_best/weights.562-0.19.hdf5') # FSA
        #model.load_weights('/storage01/users/user/ny_test/FDPA/weights.438-0.19.hdf5')
        
        for layer in model.layers:
            layer.trainable = False

        for layer in model.layers[-3:]:
            layer.trainable = True
        
        # Compile the model with the optimizer, loss function
        # and the metrics.
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE, decay_steps=5000, decay_rate=0.98, staircase=True
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                #keras.metrics.Precision(),
                #keras.metrics.Recall(),
                #keras.metrics.AUC()
                #keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )
        #earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint("/storage01/users/user/ny_test/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', save_best_only=True, mode='auto')
        # Train the model.
        _ = model.fit(trainloader, epochs=EPOCHS, validation_data=testloader, shuffle=True, callbacks=[checkpoint],) #, checkpoint, earlystopping

        _, accuracy = model.evaluate(testloader)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        #print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        return model


model = run_experiment()
#model.save('vivit_best_model.h5')