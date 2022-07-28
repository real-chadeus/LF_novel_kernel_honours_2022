from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

class LFSEBlock(tf.keras.Model):
    '''
    Light field squeeze-and-excitation block.
    Comes after convolution and pooling blocks
    '''
    def __init__(self, n_filters, filter_size):
        self.n_filters = n_filters
        self.filter_size = filter_size
        super(LFSEBlock, self).__init__(name='light_field_se_block')
    
    def call(self, input_tensor, training=True):
        shape = tf.shape(input_tensor) 
        height = shape[2]
        width = shape[4]
        print(f'h {height}, w {width}')
        for i in range(self.n_filters):
            f_map = input_tensor[i]
            z_c = (1/(height * width)) * f_map.sum(axis=(1,3)) # squeeze across the angular axes
        return tf.nn.linear(z_c)

def LF_conv_block(inputs, n_filters=4, 
                    filter_size=(3,3), n_sais=49, 
                    stride=2, img_shape=(7,420,7,420,3),
                    n_lfi=1): 
    '''
    Convolution block for light field images.
    Does convolution depthwise on each SAI individually
    '''
    n_ang = int(np.sqrt(n_sais)) # angular dimension size
    fmaps = [] # feature maps
    X = inputs
    for i in range(n_lfi):
        if len(X.shape) == 6:
            X1 = X[i,:,:,:,:]
        else:
            X1 = X
        X1 = tf.reshape(X1, (n_sais, img_shape[1], img_shape[3], img_shape[-1]))
        X1 = layers.DepthwiseConv2D(filter_size, strides=1, padding='same', input_shape=X.shape[2:], activation='relu')(X1)  
        X1 = tf.reshape(X1, (n_ang, img_shape[1], n_ang, img_shape[3], img_shape[-1]))
        fmaps.append(X1)
    X = tf.squeeze(tf.stack(fmaps, axis=1))
    X = layers.Conv3D(n_filters, 1, padding='same', activation='relu')(X)
    return X

def build_model(input_shape, summary=True):
    '''
    build the model
    '''
    inputs = layers.Input(shape=input_shape, name='lfse_model_input')
    X = layers.Conv3D(filters=3, kernel_size=(3,3,3), padding='same')(inputs) 
    X = tf.nn.relu(X)
    
    X = LF_conv_block(X, n_filters=3, filter_size=(3,3))
    X = layers.MaxPooling3D(pool_size=(2,1,2), padding='same')(X)
    X = LF_conv_block(X, n_filters=6, filter_size=(3,3), img_shape=X.shape) 
    X = LF_conv_block(X, n_filters=6, filter_size=(3,3), img_shape=X.shape)
    X = layers.MaxPooling3D(pool_size=(2,1,2), padding='same')(X)
    X = LF_conv_block(X, n_filters=12, filter_size=(3,3), img_shape=X.shape) 
    X = LF_conv_block(X, n_filters=12, filter_size=(3,3), img_shape=X.shape)
    X = layers.MaxPooling3D(pool_size=(2,1,2), padding='same')(X)
    X = LF_conv_block(X, n_filters=24, filter_size=(3,3), img_shape=X.shape) 
    print('finish')
    
    X = LFSEBlock(n_filters=24, filter_size=(3,3))(X)
    
    if summary:
        model.summary()

    return model












