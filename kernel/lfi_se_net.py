from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf


class LFSEBlock(tf.keras.Model):
    '''
    Light field squeeze-and-excitation block.
    Comes after convolution and pooling blocks
    '''
    def __init__(self, kernel_size, filters):
        super(LFSEBlock).__init__(name='light_field_se_block')
    
    def call(self, input_tensor, training=True):
        shape = tf.shape(input_tensor) 
        height = shape[2]
        width = shape[4]
        n_maps = shape[0] # number of feature maps
        for i in range(n_maps):
            f_map = input_tensor[i]
            z_c = (1/(height * width)) * f_map.sum(axis=(1,2,3,4))
        return tf.nn.linear(X)

def LF_conv_block(inputs, n_filters=4, filter_size=(3,3), n_sais=49, stride=2): 
    '''
    Convolution block for light field images.
    Does convolution depthwise on each SAI individually
    '''
    X = layers.Conv3D(n_filters, 1, padding='same', activation='relu')(inputs) 
    fmaps = [] # feature maps for each subview
    for i in range(n_sais):
        X1 = X[:,i,:,:,:]
        X1 = layers.DepthwiseConv2D(filter_size, strides=1, padding='same', input_shape=X.shape[2:], activation='relu')(X1)  
        fmaps.append(X1)
    X = tf.stack(fmaps, axis=1)
    X = layers.Conv3D(n_filters, 1, padding='same', activation='relu')(X)
    return X

def build_model(input_shape, summary=True):
    '''
    build the model
    '''
    X = layers.Conv3D(filters=3, kernel_size=(3,3,3), strides=(1,2,2), padding='same')(inputs) 
    X = tf.nn.relu(X)
    
    X = LF_conv_block(X, n_filters=3, filter_size=(3,3))
    X = layers.MaxPooling3D(pool_size=(1,2,2))
    X = LF_conv_block(X, n_filters=6, filter_size=(3,3)) 
    X = LF_conv_block(X, n_filters=6, filter_size=(3,3))
    X = layers.MaxPooling3D(pool_size=(1,2,2))
    X = LF_conv_block(X, n_filters=12, filter_size=(3,3)) 
    X = LF_conv_block(X, n_filters=12, filter_size=(3,3))
    X = layers.MaxPooling3D(pool_size=(1,2,2))
    X = LF_conv_block(X, n_filters=24, filter_size=(3,3)) 
    
    X = LFSEBlock()
    
    if summary:
        model.summary()

    return model









