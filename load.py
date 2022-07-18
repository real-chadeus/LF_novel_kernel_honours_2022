import numpy as np
import os
import PIL
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import sys, glob, os, random
import pandas as pd
import scipy.io as sio
import preprocessing.flatten
import preprocessing.hci_dataset_tools.file_io as hci_io
print('tensorflow version: ', tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_path = '../../datasets'
hci_boxes = '/hci_dataset/training/boxes/'
hci_boxes_stacked = '/hci_dataset/training/boxes/stacked/'


def load_hci(num_imgs=1, 
                 read_dirs=[data_path+hci_boxes],
                 img_shape = (7,512,7,512,3)):
    '''
    load images and depth maps into tensorflow dataset (from HCI) 
    '''
    labels = []
    img_set = []
    for i in range(num_imgs):
        img = Image.open(read_dirs[i] + '/stacked/stacked.png')
        img = np.asarray(img)
        img = img.reshape(img_shape, order='F')
        img_set.append(img)
        # read depth map as labels
        depth = hci_io.read_depth(data_path + hci_boxes)
        labels.append(depth)
    img_set = np.asarray(img_set)
    labels = np.asarray(labels)
    print(img_set.shape)
    dataset = tf.data.Dataset.from_tensor_slices((img_set, labels))
    return dataset


def load_sintel(num_imgs=1, 
                 read_dirs=[data_path+hci_boxes],
                 img_shape = (7,512,7,512,3)):
    '''
    load images and disparity maps from Sintel dataset.
    Also maps disparity maps to depth maps 
    '''
    disparities = []
    for i in range(num_imgs):
        img = Image.open(read_dirs[i] + '/stacked/stacked.png')
        img = np.asarray(img)
        img = img.reshape(img_shape, order='F')
        img_set.append(img)
        # read disparity maps
        disp = sintel_io.read_disp(data_path + hci_boxes)
        disparities.append(disp)
    


def load_dataset(num_imgs=1, 
                 read_dirs=[data_path+hci_boxes],
                 img_shape = (7,512,7,512,3)):
    '''
    load images and depth maps into tensorflow dataset (from HCI) 
    '''
    labels = []
    img_set = []
    for i in range(num_imgs):
        img = Image.open(read_dirs[i] + '/stacked/stacked.png')
        img = np.asarray(img)
        img = img.reshape(img_shape, order='F')
        img_set.append(img)
        # read depth map as labels
        depth = hci_io.read_depth(data_path + hci_boxes)
        labels.append(depth)
    img_set = np.asarray(img_set)
    labels = np.asarray(labels)
    print(img_set.shape)
    dataset = tf.data.Dataset.from_tensor_slices((img_set, labels))
    return dataset



#if __name__ == "__main__":
#    data_path = '../../datasets'
#    disparity = hci_io.read_disparity(data_path + hci_boxes)
#    print('disparity shape: \n', disparity.shape)
#    depth = hci_io.read_depth(data_path + hci_boxes)
#    print('depth shape: \n', depth.shape)
#    
#    dataset = load_dataset()
#    for elem in dataset:
#        print(elem.numpy().shape)
#














