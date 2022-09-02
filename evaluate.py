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

load_path = 'saved_models/'
input_shape = (9,436,9,436,3)
hci = load_data.load_hci(img_shape=input_shape, use_disp=True)
model = keras.models.load_model(load_path + 'model4', custom_objects={'BadPix': BadPix})
metrics = model.evaluate(x=hci[0], y=hci[1], batch_size=1, workers=4)
plots.plot_mse('model4')
plots.plot_badpix('model4')
print(metrics)













