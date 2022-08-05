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
import preprocessing.flatten as flatten
import preprocessing.hci_dataset_tools.file_io as hci_io

data_path = '../../datasets'

def load_hci(img_shape = (7,512,7,512,3)):
    '''
    load images and depth maps into tensorflow dataset (from HCI) 
    '''
    hci_folder = [d for d in os.scandir(data_path + '/hci_dataset/') if d.is_dir()]
    labels = []
    img_set = []

    for s in hci_folder:
        sub_dir = s.path
        hci_r_dirs = [d for d in os.scandir(sub_dir) if d.is_dir()]
        for d in hci_r_dirs:
            r_dir = d.path
            if 'test' in r_dir:
                continue
            img = Image.open(r_dir + '/stacked/stacked.png')
            img = np.asarray(img)
            img = img.reshape(img_shape, order='F')
            img_set.append(img)
            # read depth map as labels
            depth = hci_io.read_depth(r_dir)
            labels.append(depth)

    img_set = np.asarray(img_set)
    labels = np.asarray(labels)
    dataset = (img_set, labels)
    #dataset = tf.data.Dataset.from_tensor_slices(dataset)
    return dataset

def load_sintel(img_shape = (7,512,7,512,3)):
    '''
    load images and disparity maps from Sintel dataset.
    Also converts disparity maps to depth maps 
    '''
    sintel_r_dirs = [d for d in os.scandir(data_path + '/Sintel_LF/Sintel_LFV_9x9_with_all_disp/') if d.is_dir()]
    img_set = []
    labels = []

    for d in sintel_r_dirs:
        r_dir = d.path + '/stacked/'
        n_frames = len([f for f in os.scandir(r_dir) if f.is_file()])//3 #number of frames in the current scene
        for i in range(n_frames):
            if i < 10:
                frame = f"00{i}"
            else:
                frame = f"0{i}"
            
            if i > 2:
                # load only the first 3 frames of each scene
                break

            img = Image.open(r_dir + frame + '_stacked.png')
            img = np.asarray(img)
            img = img.reshape(img_shape, order='F')
            img_set.append(img)
            # read disparity maps
            disp = np.load(r_dir + frame + '_center.npy')
            depth = 0.01 * 1 / disp
            labels.append(depth)
            print('loaded image {}'.format(r_dir + frame + '_stacked.png'))

    img_set = np.asarray(img_set)
    labels = np.asarray(labels)
    print(':)', img_set.shape)
    dataset = (img_set, labels)
    #dataset = tf.data.Dataset.from_tensor_slices(dataset)
    return dataset
















