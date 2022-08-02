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
        r = 4 # reductionr ratio
        C = self.n_filters
        shape = tf.shape(input_tensor) 
        height = shape[1]
        width = shape[3]
        W1 = tf.cast(tf.convert_to_tensor(np.ones((C, C//r))), dtype=tf.float32) 
        W2 = tf.cast(tf.convert_to_tensor(np.ones((C//r, C))), dtype=tf.float32)
        z = []
        f_maps = [] # original feature maps
        s = []
        for i in range(self.n_filters):
            f_map = input_tensor[:,:,:,:,i]
            f_maps.append(f_map)
            z_c = tf.cast(1/(height * width), dtype=tf.float32) * tf.math.reduce_sum(f_map, axis=None) # squeeze 
            z.append(z_c)
        # excitation
        z = tf.reshape(z, (self.n_filters,1))
        g = tf.nn.relu(W1 * z)
        g = tf.reshape(g, (g.shape[1], g.shape[0]))
        s = tf.math.reduce_sum(tf.math.sigmoid(W2 * g), axis=0) 
        result = s * input_tensor
        # final sum over the angular axes 
        result = tf.math.reduce_sum(result, axis=(0,2))

        return tf.nn.relu(result)

def LF_conv_block(inputs, n_filters=4, 
                    filter_size=(3,3), n_sais=49, 
                    stride=2, img_shape=(7,512,7,512,3),
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

def build_model(input_shape, output_shape=420, summary=True):
    '''
    build the model
    param output_shape: size of the 2D depth map. default 420
    '''
    # initial input and convolution + layer normalization
    inputs = layers.Input(shape=input_shape, name='lfse_model_input')
    X = layers.Dense(3)(inputs)
    X = layers.LayerNormalization()(X)
    X = layers.Conv3D(filters=3, kernel_size=(3,3,3), padding='same')(X) 
    X = tf.nn.relu(X)
    
    X = LF_conv_block(X, n_filters=3, filter_size=(3,3))
    X = LF_conv_block(X, n_filters=6, filter_size=(3,3), img_shape=X.shape) 
    X = LF_conv_block(X, n_filters=6, filter_size=(3,3), img_shape=X.shape)
    X = LF_conv_block(X, n_filters=12, filter_size=(3,3), img_shape=X.shape) 
    X = LF_conv_block(X, n_filters=12, filter_size=(3,3), img_shape=X.shape)
    X = LF_conv_block(X, n_filters=24, filter_size=(3,3), img_shape=X.shape) 
    X = LF_conv_block(X, n_filters=24, filter_size=(3,3), img_shape=X.shape)
    X = LFSEBlock(n_filters=24, filter_size=(3,3))(X)
    X = layers.RandomFlip()(X)
    
    X = layers.Dense(512, activation='relu')(X)
    X = layers.Dense(1024, activation='relu')(X)
    X = layers.Dense(2048, activation='relu')(X)
    #X = layers.Dense(4096, activation='relu')(X)
    X = tf.squeeze(layers.Dense(1, activation='sigmoid')(X))
    
    model = models.Model(inputs=inputs, outputs=X)   
 
    if summary:
        model.summary()

    return model












