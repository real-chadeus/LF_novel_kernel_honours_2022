import pathlib, datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import load
import preprocessing.flatten
import preprocessing.hci_dataset_tools.file_io as hci_io
import kernel.lfi_se_net as se_net
import tensorflow.keras.losses as losses
import argparse
import plots
from custom_metrics import BadPix

load_path = 'models/'
input_shape = (3,436,3,436,3)
hci = load.load_hci(img_shape=input_shape)
model = keras.models.load_model(load_path + 'model3', custom_objects={'BadPix': BadPix})
metrics = model.evaluate(x=hci[0], y=hci[1], batch_size=1, workers=4)
plots.plot_mse('model3')
plots.plot_badpix('model3')
print(metrics)













