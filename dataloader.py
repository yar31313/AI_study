import os
from data_preprocessing  import *

from easydict import EasyDict
import json

import tensorflow as tf
from tqdm import tqdm

class JsonConfigFileManager:
    def __init__(self, file_path):
        self.values = EasyDict()
        if file_path:
            self.file_path = file_path # 파일경로 저장
            self.reload()

    def reload(self):
        self.clear()
        if self.file_path:
            with open(self.file_path, 'r') as f:
                self.values.update(json.load(f))

    def clear(self):
        self.values.clear()
                
    def update(self, in_dict):
        for (k1, v1) in in_dict.items():
            if isinstance(v1, dict):
                for (k2, v2) in v1.items():
                    if isinstance(v2, dict):
                        for (k3, v3) in v2.items():
                            self.values[k1][k2][k3] = v3
                    else:
                        self.values[k1][k2] = v2
            else:
                self.values[k1] = v1     
            
    def export(self, save_file_name):
        if save_file_name:
            with open(save_file_name, 'w') as f:
                json.dump(dict(self.values), f)

def dataloader(df,data_path,input_shape):
  img_concat = []
  label_concat = []
  for i in tqdm(range(len(df))):
      folder_name = df['patient_ID'].iloc[i]

      patient_path = os.path.join(data_path,folder_name)

      dicom_series = readTotalVolume(patient_path)
      dicom_series = resample_Total(dicom_series)
      dicom_series = resize_Total(dicom_series, input_shape[1],input_shape[2],input_shape[0]*2)
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
    # videos: np.ndarray,
    # labels: np.ndarray,
    # loader_type: str = "train",
    # batch_size: int = 4,
    # ):
    videos,
    labels,
    loader_type,
    batch_size = 1,
    ):
    """Utility function to prepare the dataloader."""
    dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

    if loader_type == "train":
        dataset = dataset.shuffle(batch_size * 2)

    dataloader = (
        dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader
