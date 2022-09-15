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
        new_disp = np.rot90(disp, k=0, axes=(d_axis1,d_axis2))
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
    
    if use_gen == False:
        return imgs, disps

def load_hci(img_shape = (9,512,9,512,3), do_augment=False, 
                predict=False, use_tf_ds=False, use_disp=True):
    '''
    load images and depth maps into tensorflow dataset (from HCI) 
    arg use_disp: use disparity maps instead of depth maps
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
            img = img.reshape((9,512,9,512,3), order='F')
            img = img.reshape((9,512,9,512,3), order='F')
            img = np.moveaxis(img, 2, 3)
            # take the red channel
            lfi = img[:,:,:,:,0]

            if predict == False:
                d_map = []
                if use_disp: 
                    d_map = np.load(r_dir + '/stacked/center_disp.npy')
                else:
                    d_map = np.load(r_dir + '/stacked/center_depth.npy')

                d_map = np.swapaxes(d_map, 0, 1)
                labels.append(d_map)

            if use_tf_ds:
                img = np.expand_dims(lfi, axis=0) # for using tf.dataset.Dataset datasets

            img_set.append(lfi)

    if predict:
        return np.asarray(img_set)

    return (np.asarray(img_set), np.asarray(labels))


def load_sintel(img_shape = (7,512,7,512,3), do_augment=True, use_tf_ds=True, use_disp=True):
    '''
    load images and disparity maps from Sintel dataset.
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
            
            #if i > 2:
            #    continue
            
            # load images
            img = Image.open(r_dir + frame + '_stacked.png')
            img = np.asarray(img)
            img = img.reshape(img_shape, order='F')

            # read + normalize disparity maps
            d_map = np.load(r_dir + frame + '_center.npy')
            if use_disp:
                d_map = d_map/np.abs(np.amax(d_map))
            else:
                # convert disp to depth
                d_map = 0.01 * 1 / d_map 
                d_map = d_map/np.amax(d_map)
            
            if do_augment:
                ds = (img, d_map)
                imgs, maps = augment(ds)
                for im in imgs:
                    if use_tf_ds:
                        im = np.expand_dims(im, axis=0) # for using tf.dataset.Dataset datasets
                    img_set.append(im)
                for d in maps:
                    labels.append(d)

            if use_tf_ds:
                img = np.expand_dims(img, axis=0) # for using tf.dataset.Dataset datasets

            img_set.append(img)
            labels.append(d_map)

    img_set = np.asarray(img_set)
    labels = np.asarray(labels)
    print('img_set shape', img_set.shape)
    print('labels shape', labels.shape)
    dataset = (img_set, labels)
    return dataset
    

def dataset_gen(augment_sintel=True, augment_hci=True,
                load_sintel=True, load_hci=True, angres=9, batch_size=16, batches=1000):
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
                    for im, m in augment(ds, img_shape=(9,512,512,9), num_flips=2, num_rot=2, num_contrast=2,
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
                if 'test' in r_dir:
                    continue
                # load images
                img = Image.open(r_dir + '/stacked/stacked.png')
                img = np.asarray(img)
                img = img.reshape((9,512,9,512,3), order='F')
                img = np.moveaxis(img, 2, 3)
                # take the red channel
                lfi = img[:,:,:,:,0]

                # load and normalize disparity maps
                d_map = np.load(r_dir + '/stacked/center_disp.npy')
                d_map = np.swapaxes(d_map, 0, 1)
                #d_map = d_map/np.abs(np.amax(d_map))

                if augment_hci:
                    ds = (lfi, d_map)
                    for im, m in augment(ds, img_shape=(9,512,512,9), num_flips=35, num_rot=35, num_contrast=35,
                                               num_noise=35, num_sat=0, num_bright=0, num_gamma=0, num_hue=0):
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












