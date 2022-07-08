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
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


data_path = '../../datasets'
hci_boxes = '/hci_dataset/training/boxes/stacked/'

def load_dataset(num_imgs=1, read_dirs=[data_path+hci_boxes]):
    '''
    by default reads the boxes 
    '''
    img_set = []
    for i in range(num_imgs):
        img = Image.open(read_dir[i] + 'stacked.png')
        img = np.asarray(img)
        img_set.append(img)
    img_set = np.asarray(img_set)
    dataset = tf.data.Dataset.from_tensor_slices(img_set)
    return dataset

def create_model():
    inputs = tf.keras.Input(shape=(7,434,7,434,3)) 
    outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)

if __name__ == "__main__":
    data_path = '../../datasets'
    dataset = load_hci()
    for elem in dataset:
        print(elem.numpy())





    








