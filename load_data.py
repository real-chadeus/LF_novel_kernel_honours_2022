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
import threading

data_path = '../../datasets'

def augment(dataset, img_shape=(81,512,512,3), num_flips=1, num_rot=1, num_contrast=1,
            num_noise=1, num_sat=1, num_bright=1, num_gamma=1, num_scale=1, 
            num_hue=1, use_gen=True):
    '''
    custom augment function
    returns: tuple (images, depth maps)
    arg dataset: tuple (image, depth map)
    '''
    img = dataset[0]
    disp = dataset[1]

    # random flip
    for i in range(num_flips):
        axes = [1,2]
        flip_axis = np.random.choice(axes)
        if flip_axis == 1:
            d_axis = 0
        else:
            d_axis = 1

        new_img = np.flip(img, axis=flip_axis)
        new_disp = np.flip(disp, axis=d_axis)
        new_img = 0.2126 * new_img[:,:,:,:,0] + 0.7152 * new_img[:,:,:,:,1] + 0.0722 * new_img[:,:,:,:,2]

        yield (new_img, new_disp)
    
    # random 90 degree rotate
    for i in range(num_rot):
        rots = [0,1,2,3]
        n_rot = np.random.choice(rots)
            
        new_img = np.rot90(img, k=n_rot, axes=(1,2))
        new_disp = np.rot90(disp, k=n_rot, axes=(0,1))
        new_img = 0.2126 * new_img[:,:,:,:,0] + 0.7152 * new_img[:,:,:,:,1] + 0.0722 * new_img[:,:,:,:,2]
        imgs.append(new_img)
        imgs.append(new_disp)
        yield (new_img, new_disp)
    
    # random noise
    for i in range(num_noise):
        noise = np.random.uniform(0, 1, size=img.shape) 
        new_img = img * noise
        new_img = 0.2126 * new_img[:,:,:,:,0] + 0.7152 * new_img[:,:,:,:,1] + 0.0722 * new_img[:,:,:,:,2]
        new_disp = disp * noise[4, :, :, 4, :]
        yield (new_img, new_disp)

    # random contrast
    for i in range(num_contrast):
        factor = np.random.uniform(0.8,1.2)
        new_img = tf.image.adjust_contrast(img, contrast_factor=factor).numpy()
        new_img = 0.2126 * new_img[:,:,:,:,0] + 0.7152 * new_img[:,:,:,:,1] + 0.0722 * new_img[:,:,:,:,2]
        new_disp = disp
        yield (new_img, new_disp)

    # random saturation
    for i in range(num_sat):
        factor = np.random.uniform(-3,3)
        new_img = tf.image.adjust_saturation(img, saturation_factor=factor).numpy() 
        new_img = 0.2126 * new_img[:,:,:,:,0] + 0.7152 * new_img[:,:,:,:,1] + 0.0722 * new_img[:,:,:,:,2]
        new_disp = disp
        yield (new_img, new_disp)

    # random brightness
    for i in range(num_bright):
        factor = np.random.uniform(-1,1)
        new_img = tf.image.adjust_brightness(img, delta=factor).numpy()
        new_img = 0.2126 * new_img[:,:,:,:,0] + 0.7152 * new_img[:,:,:,:,1] + 0.0722 * new_img[:,:,:,:,2]
        new_disp = disp
        yield (new_img, new_disp)

    # random gamma
    for i in range(num_gamma):
        factor = np.random.uniform(0.8, 1.2)
        new_img = tf.image.adjust_gamma(img, gamma=factor).numpy()
        new_disp = disp
        new_img = 0.2126 * new_img[:,:,:,:,0] + 0.7152 * new_img[:,:,:,:,1] + 0.0722 * new_img[:,:,:,:,2]
        yield (new_img, new_disp)

    # random hue
    for i in range(num_hue):
        factor = np.random.uniform(-1,1)
        new_img = tf.image.adjust_hue(img, delta=factor).numpy()
        new_img = 0.2126 * new_img[:,:,:,:,0] + 0.7152 * new_img[:,:,:,:,1] + 0.0722 * new_img[:,:,:,:,2]
        new_disp = disp
        yield (new_img, new_disp)

    # random scale
    for i in range(num_scale):
        factor = np.random.uniform(0.25, 1)
        new_img = img * factor
        new_disp = disp * factor
        new_img = 0.2126 * new_img[:,:,:,:,0] + 0.7152 * new_img[:,:,:,:,1] + 0.0722 * new_img[:,:,:,:,2]
        yield (new_img, new_disp)
    

def random_crop(img, disp):
    xy_range = np.arange(0, 16)
    x = np.random.choice(xy_range)
    y = np.random.choice(xy_range)
    crop_img = img[:, 32*x:32*(x+1), 32*y:32*(y+1), :]
    crop_map = disp[32*x:32*(x+1),32*y:32*(y+1)]
    return (crop_img, crop_map)

def dataset_gen(augment_sintel=True, augment_hci=True, crop=True, window_size=32,
                load_sintel=True, load_hci=True, angres=9, batch_size=16, batches=1000,
                train=True, validation=False, test=False, full_size=False):
    '''
    yields images and disparity maps from both datasets as a generator (reduces memory usage).
    Loads in order of Sintel -> HCI
    For training only.
    '''

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

                    d_map = np.load(r_dir + '/stacked/center_disp.npy')
                    d_map = np.swapaxes(d_map, 0, 1)

                    crop_img, crop_map = random_crop(img, d_map) 
                    if augment_hci:
                        ds = (crop_img, crop_map)
                        for im, m in augment(ds, img_shape=(9,32,32,9), num_flips=0, num_rot=0, num_scale=1, num_contrast=1,
                                                   num_noise=0, num_sat=0, num_bright=0, num_gamma=1, num_hue=1):
                            if len(imgs) < batch_size:
                                imgs.append(im)
                                maps.append(m) 
                            if len(imgs) == batch_size:
                                yield (imgs, maps)
                                imgs = []
                                maps = []
                    if len(imgs) < batch_size:
                        crop_img = 0.2126 * crop_img[:,:,:,:,0] + 0.7152 * crop_img[:,:,:,:,1] + 0.0722 * crop_img[:,:,:,:,2]
                        imgs.append(crop_img)
                        maps.append(crop_map) 
                    if len(imgs) == batch_size:
                        yield (imgs, maps)
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

                    if full_size:
                        if test:
                            if len(imgs) < batch_size:
                                imgs.append(lfi)
                            if len(imgs) == batch_size:
                                yield imgs
                                imgs = []
                        else:
                            imgs.append(lfi)
                            maps.append(d_map) 
                            yield (imgs, maps)
                            imgs = []
                            maps = []
                    else:
                        for x in range(16):
                            for y in range(16): 
                                crop_img = lfi[:, 32*x:32*(x+1), 32*y:32*(y+1), :]
                                crop_map = d_map[32*x:32*(x+1),32*y:32*(y+1)]
                                if len(imgs) < batch_size:
                                    imgs.append(crop_img)
                                    maps.append(crop_map) 
                                if len(imgs) == batch_size:
                                    yield (imgs, maps)
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
                    
                    if full_size:
                        if len(imgs) < batch_size:
                            imgs.append(lfi)
                        if len(imgs) == batch_size:
                            yield imgs
                            imgs = []
                    else:
                        for x in range(16):
                            for y in range(16): 
                                crop_img = lfi[:, 32*x:32*(x+1), 32*y:32*(y+1), :]
                                if len(imgs) < batch_size:
                                    imgs.append(crop_img)
                                if len(imgs) == batch_size:
                                    yield imgs
                                    imgs = []


class ThreadsafeIter:
    """
    uses mutex to serialize
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe(f):
    """
    decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return ThreadsafeIter(f(*a, **kw))
    return g

@threadsafe
def multi_input(dataset, angres=9, test=False):
    while 1:
        for data in dataset:
            if test:
                img_set = data
            else:
                img_set = data[0]

            sai_list = []
            for i in range(angres):
                for k in range(angres):
                    sai_list.append(img_set[:,i,:,:,k])

            if test:
                yield sai_list,
            else:
                target = data[1]
                yield sai_list, target












