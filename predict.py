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

load_path = 'models/'

input_shape = (3,436,3,436,3)
#model = se_net.build_model(input_shape=input_shape, summary=True, n_sais=9)
hci = load.load_hci(img_shape=input_shape, predict=True)
model = keras.models.load_model(load_path + 'model0')
predictions = model.predict(hci, batch_size=1, workers=4)
print(predictions)














