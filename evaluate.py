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
batch_size=1
gen = load_data.dataset_gen
hci = tf.data.Dataset.from_generator(gen, 
                 args=(False, False, True, 32, False, 
                       True, 9, batch_size, 1000, False, True, False),
                        output_signature=(tf.TensorSpec(shape=(batch_size,) + input_shape, dtype=tf.float32),
                                          tf.TensorSpec(shape=(batch_size,) + (input_shape[1], input_shape[2]), dtype=tf.float32)))
model = net.build_model(input_shape=input_shape, summary=True, 
                                n_sais=81, batch_size=1)
custom_metrics = {'BadPix7': BadPix(threshold=0.07), 'BadPix3': BadPix(threshold=0.03), 'BadPix1': BadPix(threshold=0.01)}
model = keras.models.load_model(load_path + model_name, custom_objects={'BadPix': BadPix})
metrics = model.evaluate(hci,  workers=8)
print(metrics)

















