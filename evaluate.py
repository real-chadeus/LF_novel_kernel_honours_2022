import pathlib, datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import load_data
import preprocessing.hci_dataset_tools.file_io as hci_io
import model.model as net
import tensorflow.keras.losses as losses
import argparse
import plots
from custom_metrics import BadPix
import functools

load_path = 'checkpoints/'
model_name = 'test/'
input_shape = (9,32,32,9)
hci = functools.partial(load_data.dataset_gen, 
                            load_sintel=False, load_hci=True, crop=False, window_size=32,
                            augment_sintel=False, augment_hci=False,
                            batch_size=1)
hci = tf.data.Dataset.from_generator(hci,
      output_signature=(tf.TensorSpec(shape=(1,) + input_shape, dtype=tf.int8),
                        tf.TensorSpec(shape=(1,) + (input_shape[1], input_shape[2]), dtype=tf.float32)))
model = net.build_model(input_shape=input_shape, summary=True, 
                                n_sais=81, batch_size=1)
custom_metrics = {'BadPix7': BadPix(threshold=0.07), 'BadPix3': BadPix(threshold=0.03), 'BadPix1': BadPix(threshold=0.01)}
model = keras.models.load_model(load_path + model_name, custom_objects={'BadPix': BadPix})
metrics = model.evaluate(hci, batch_size=1, workers=4)
print(metrics)

















