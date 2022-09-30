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
import preprocessing.hci_dataset_tools.file_io as hci_io

data_path = '../../datasets'

def augment(dataset, img_shape=(81,512,512,3), num_flips=1, num_rot=1, num_contrast=1,
            num_noise=1, num_sat=1, num_bright=1, num_gamma=1, 
            num_hue=1, use_gen=True):
    '''
    custom augment function
    returns: tuple (images, depth maps)
    arg dataset: tuple (image, depth map)
    '''
    img = dataset[0]
    disp = dataset[1]
    imgs = [] 
    disps = []

    # random flip
    for i in range(num_flips):
        axes = [1,2]
        flip_axis = np.random.choice(axes)
        if flip_axis == 2:
            d_axis = 1
        else:
            d_axis = 0

        new_img = np.flip(img, axis=flip_axis)
        new_disp = np.flip(disp, axis=d_axis)
        if use_gen:
            yield (new_img, new_disp)
        else:
            imgs.append(new_img)
            imgs.append(new_disp)
    
    # random 90 degree rotate
    for i in range(num_rot):
        axes = [1,2]
        axis1 = np.random.choice(axes)
        if axis1 == 2:
            axis2 = 1
            d_axis1 = 1
            d_axis2 = 0
        else:
            axis2 = 2
            d_axis1 = 0
            d_axis2 = 1
            
        new_img = np.rot90(img, k=1, axes=(axis1,axis2))
        new_disp = np.rot90(disp, k=1, axes=(d_axis1,d_axis2))
        imgs.append(new_img)
        imgs.append(new_disp)
        if use_gen:
            yield (new_img, new_disp)
        else:
            imgs.append(new_img)
            imgs.append(new_disp)
    
    # Gaussian noise
    for i in range(num_noise):
        noise = np.random.normal(loc=0.0, scale=3, size=img.shape) 
        new_img = img + noise
        new_disp = disp
        if use_gen:
            yield (new_img, new_disp)
        else:
            imgs.append(new_img)
            imgs.append(new_disp)

    # random contrast
    for i in range(num_contrast):
        factor = np.random.uniform(-3,3)
        new_img = tf.image.adjust_contrast(img, contrast_factor=factor).numpy()
        new_disp = disp
        if use_gen:
            yield (new_img, new_disp)
        else:
            imgs.append(new_img)
            imgs.append(new_disp)

    # random saturation
    for i in range(num_sat):
        factor = np.random.uniform(-3,3)
        new_img = tf.image.adjust_saturation(img, saturation_factor=factor).numpy() 
        new_disp = disp
        if use_gen:
            yield (new_img, new_disp)
        else:
            imgs.append(new_img)
            imgs.append(new_disp)

    # random brightness
    for i in range(num_bright):
        factor = np.random.uniform(-1,1)
        new_img = tf.image.adjust_brightness(img, delta=factor).numpy()
        new_disp = disp
        if use_gen:
            yield (new_img, new_disp)
        else:
            imgs.append(new_img)
            imgs.append(new_disp)

    # random gamma
    for i in range(num_gamma):
        factor = np.random.uniform(0,3)
        new_img = tf.image.adjust_gamma(img, gamma=factor).numpy()
        new_disp = disp
        if use_gen:
            yield (new_img, new_disp)
        else:
            imgs.append(new_img)
            imgs.append(new_disp)

    # random hue
    for i in range(num_hue):
        factor = np.random.uniform(-1,1)
        new_img = tf.image.adjust_hue(img, delta=factor).numpy()
        new_disp = disp
        if use_gen:
            #new_img = np.expand_dims(new_img, axis=0) # for using tf.dataset.Dataset datasets
            yield (new_img, new_disp)
        else:
            imgs.append(new_img)
            imgs.append(new_disp)
    

def random_crop(img, disp):
    xy_range = np.arange(0, 16)
    x = np.random.choice(xy_range)
    y = np.random.choice(xy_range)
    crop_img = img[:, 32*x:32*(x+1), 32*y:32*(y+1), :]
    crop_map = disp[32*x:32*(x+1),32*y:32*(y+1)]
    return (crop_img, crop_map)


def dataset_gen(augment_sintel=True, augment_hci=True, crop=True, window_size=32,
                load_sintel=True, load_hci=True, angres=9, batch_size=16, batches=1000,
                train=True, validation=False, test=False):
    '''
    yields images and disparity maps from both datasets as a generator (reduces memory usage).
    Loads in order of Sintel -> HCI
    For training only.
    '''
    if load_sintel:
        imgs = []
        maps = []
        sintel_r_dirs = [d for d in os.scandir(data_path + '/Sintel_LF/Sintel_LFV_9x9_with_all_disp/') if d.is_dir()]
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
                img = img.reshape((9,512,9,512,3), order='F')
                img = img.reshape((angres, h, angres, w, 3), order='F')
                img = np.moveaxis(img, 2, 3)
                # take the red channel
                lfi = img[:,:,:,:,0]

                # read + normalize disparity maps
                d_map = np.load(r_dir + frame + '_center.npy')
                d_map = np.swapaxes(d_map, 0, 1)
                
                if augment_sintel:
                    ds = (lfi, d_map)
                    for im, m in augment(ds, img_shape=(9,32,32,9), num_flips=2, num_rot=2, num_contrast=2,
                                               num_noise=2, num_sat=2, num_bright=2, num_gamma=2, num_hue=0):
                        if len(imgs) < batch_size:
                            imgs.append(im)
                            maps.append(m) 

                        if len(imgs) == batch_size:
                            yield (np.asarray(imgs), np.asarray(maps))
                            imgs = []
                            maps = []

                if len(imgs) == batch_size:
                    yield (np.asarray(imgs), np.asarray(maps))
                    imgs = []
                    maps = []

                imgs.append(lfi)
                maps.append(d_map)


    if load_hci:
        imgs = []
        maps = []
        
        hci_folder = [d for d in os.scandir(data_path + '/hci_dataset/') if d.is_dir()]
        for s in hci_folder:
            sub_dir = s.path
            hci_r_dirs = [d for d in os.scandir(sub_dir) if d.is_dir()]
            for d in hci_r_dirs:
                r_dir = d.path
                if train:
                    if 'additional' not in r_dir:
                        continue
                    img = Image.open(r_dir + '/stacked/stacked.png')
                    img = np.asarray(img)
                    img = img.reshape((9,512,9,512,3), order='F')
                    img = np.moveaxis(img, 2, 3)
                    img = img/255
                    lfi = 0.2126 * img[:,:,:,:,0] + 0.7152 * img[:,:,:,:,1] + 0.0722 * img[:,:,:,:,2]

                    d_map = np.load(r_dir + '/stacked/center_disp.npy')
                    d_map = np.swapaxes(d_map, 0, 1)

                    if augment_hci:
                        ds = (lfi, d_map)
                        for im, m in augment(ds, img_shape=(9,512,512,9), num_flips=150, num_rot=150, num_contrast=0,
                                                   num_noise=0, num_sat=0, num_bright=0, num_gamma=0, num_hue=0):
                            if len(imgs) < batch_size:
                                crop_img, crop_map = random_crop(im, m) 
                                imgs.append(crop_img)
                                maps.append(crop_map) 
                            if len(imgs) == batch_size:
                                yield (np.asarray(imgs), np.asarray(maps))
                                imgs = []
                                maps = []

                    if len(imgs) < batch_size:
                        crop_img, crop_map = random_crop(lfi, d_map) 
                        imgs.append(crop_img)
                        maps.append(crop_map) 
                    if len(imgs) == batch_size:
                        yield (np.asarray(imgs), np.asarray(maps))
                        imgs = []
                        maps = []

                if validation:
                    if 'stratified' not in r_dir and 'training' not in r_dir:
                        continue
                    img = Image.open(r_dir + '/stacked/stacked.png')
                    img = np.asarray(img)
                    img = img.reshape((9,512,9,512,3), order='F')
                    img = np.moveaxis(img, 2, 3)
                    img = img/255
                    lfi = 0.2126 * img[:,:,:,:,0] + 0.7152 * img[:,:,:,:,1] + 0.0722 * img[:,:,:,:,2]

                    d_map = np.load(r_dir + '/stacked/center_disp.npy')
                    d_map = np.swapaxes(d_map, 0, 1)

                    for x in range(16):
                        for y in range(16): 
                            crop_img = lfi[:, 32*x:32*(x+1), 32*y:32*(y+1), :]
                            crop_map = d_map[32*x:32*(x+1),32*y:32*(y+1)]
                            if len(imgs) < batch_size:
                                imgs.append(crop_img)
                                maps.append(crop_map) 
                            if len(imgs) == batch_size:
                                yield (np.asarray(imgs), np.asarray(maps))
                                imgs = []
                                maps = []
                if test:
                    if 'test' not in r_dir:
                        continue
                    img = Image.open(r_dir + '/stacked/stacked.png')
                    img = np.asarray(img)
                    img = img.reshape((9,512,9,512,3), order='F')
                    img = np.moveaxis(img, 2, 3)
                    img = img/255
                    lfi = 0.2126 * img[:,:,:,:,0] + 0.7152 * img[:,:,:,:,1] + 0.0722 * img[:,:,:,:,2]

                    for x in range(16):
                        for y in range(16): 
                            crop_img = lfi[:, 32*x:32*(x+1), 32*y:32*(y+1), :]
                            crop_map = d_map[32*x:32*(x+1),32*y:32*(y+1)]
                            if len(imgs) < batch_size:
                                imgs.append(crop_img)
                                maps.append(crop_map) 
                            if len(imgs) == batch_size:
                                yield (np.asarray(imgs), np.asarray(maps))
                                imgs = []
                                maps = []













