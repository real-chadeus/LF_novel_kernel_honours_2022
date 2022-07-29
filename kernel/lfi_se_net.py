from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.models as models
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
        height = shape[1]
        width = shape[3]
        z = [] 
        for i in range(self.n_filters):
            f_map = input_tensor[:,:,:,:,i]
            print(f_map.shape)
            z_c = tf.cast(1/(height * width), dtype=tf.float32) * tf.math.reduce_sum(f_map, axis=(1,3)) # squeeze across the angular axes
            z.append(z_c)
        
        return tf.nn.relu(z)

def LF_conv_block(inputs, n_filters=4, 
                    filter_size=(3,3), n_sais=49, 
                    stride=2, img_shape=(7,420,7,420,3),
                    n_lfi=1): 
    '''
    Simple convolution block for light field images.
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
    X = LFSEBlock(n_filters=12, filter_size=(3,3))(X)
    
    X = layers.Flatten()(X)
    
    model = models.Model(inputs=inputs, outputs=X)   
 
    if summary:
        model.summary()

    return model












