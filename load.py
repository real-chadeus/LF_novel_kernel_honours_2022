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

def augment(dataset):
    '''
    augment function
    returns: tuple (images, depth maps)
    arg dataset: tuple (image, depth map)
    '''
    # img of shape (x,y,x,y,3) where x is the angular dim and y is the spatial dim
    # flips and rotates 90 degrees
    img = dataset[0]
    depth = dataset[1]

    # flip
    new_img1 = np.flip(img, axis=1)
    new_depth1 = np.flip(depth, axis=0)
    new_img2 = np.flip(img, axis=3)
    new_depth2 = np.flip(depth, axis=1)
    new_img3 = np.flip(img, axis=(1,3))
    new_depth3 = np.flip(depth, axis=(0,1))
    
    # rotate
    new_img4 = np.rot90(img, k=1, axes=(1,3))
    new_depth4 = np.rot90(depth, k=0, axes=(0,1))
    new_img5 = np.rot90(img, k=1, axes=(3,1))
    new_depth5 = np.rot90(depth, k=0, axes=(1,0))
   
    imgs = [new_img1, new_img2, new_img3, new_img4, new_img5] 
    depths = [new_depth1, new_depth2, new_depth3, new_depth4, new_depth5]
    
    return imgs, depths

def load_hci(img_shape = (7,512,7,512,3), predict=False):
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
            if 'test' in r_dir and predict == False:
                continue
            # load + normalize
            img = Image.open(r_dir + '/stacked/stacked.png')
            img = np.asarray(img)
            img = img.reshape(img_shape, order='F')
            img_set.append(img)
            # read depth map as labels
            if predict == False:
                depth = np.load(r_dir + '/stacked/center.npy')
                depth = depth/np.amax(depth)
                labels.append(depth)

    if predict:
        dataset = img_set
        return dataset
    img_set = np.asarray(img_set)
    labels = np.asarray(labels)
    dataset = (img_set, labels)
    #dataset = tf.data.Dataset.from_tensor_slices(dataset)
    return dataset

def load_sintel(img_shape = (7,512,7,512,3), do_augment=True, use_tf_ds=True):
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

            # load images
            img = Image.open(r_dir + frame + '_stacked.png')
            img = np.asarray(img)
            img = img.reshape(img_shape, order='F')

            # read + normalize disparity maps
            disp = np.load(r_dir + frame + '_center.npy')
            depth = 0.01 * 1 / disp 
            depth = depth/np.amax(depth)
            
            if do_augment:
                dataset = (img, depth)
                imgs, depths = augment(dataset)
                for im in imgs:
                    if use_tf_ds:
                        im = np.expand_dims(im, axis=0) # for using tf.dataset.Dataset datasets
                    img_set.append(im)
                for depth in depths:
                    labels.append(depth)

            if use_tf_ds:
                img = np.expand_dims(img, axis=0) # for using tf.dataset.Dataset datasets

            img_set.append(img)
            labels.append(depth)
            
            # diagnostics
            #print('loaded image {}'.format(r_dir + frame + '_stacked.png'))
            #if i % 20 == 0:
            #    pr = Image.fromarray(img[1,:,1,:])
            #    pr.show()

    img_set = np.asarray(img_set)
    labels = np.asarray(labels)
    print('img_set shape', img_set.shape)
    print('labels shape', labels.shape)
    dataset = (img_set, labels)
    return dataset

     
    














