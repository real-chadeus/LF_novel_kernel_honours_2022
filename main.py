import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys, glob, os, random
import pandas as pd
import scipy.io as sio
import flatten
print('tensorflow version: ', tf.__version__)


data_path = '../../datasets'
read_dir=data_path + '/hci_dataset/training/boxes/stacked/'

def load_hci():
    img = Image.open(read_dir + 'stacked.png')
    img = np.asarray(img)
    dataset = tf.data.Dataset.from_tensors(img)
    return dataset

def create_model():
    inputs = tf.keras.Input(shape=(7,434,7,434,3))
    

if __name__ == "__main__":
    data_path = '../../datasets'
    dataset = load_hci()
    for elem in dataset:
        print(elem.numpy())





    








