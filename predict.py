import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys, glob, os, random
import pandas as pd
import scipy.io as sio
import kernel.lfi_se_net as se_net
import preprocessing.flatten as flatten
import preprocessing.hci_dataset_tools.file_io as hci_io
import load

load_path = 'models/'

input_shape = (3,436,3,436,3)
hci = functools.partial(load_data.dataset_gen, 
                            load_sintel=False, load_hci=True, crop=False, window_size=32,
                            augment_sintel=False, augment_hci=False,
                            batch_size=1)
hci = tf.data.Dataset.from_generator(hci,
      output_signature=(tf.TensorSpec(shape=(1,) + input_shape, dtype=tf.int8),
                        tf.TensorSpec(shape=(1,) + (input_shape[1], input_shape[2]), dtype=tf.float32)))
model = keras.models.load_model(load_path + 'model0')
predictions = model.predict(hci, batch_size=1, workers=4)

for i in range(predictions.shape[0]):
    print(predcitions[i].shape)
    np.save(f'predictions/pred_{i}', predictions[i])














