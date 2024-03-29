from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.models as models
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import einops
import matplotlib.pyplot as plt

def LF_conv_block(inputs, n_filters=4, 
                    filter_size=(3,3), n_sais=81, 
                    stride=2, img_shape=(7,512,7,512,3)): 
    '''
    Simple convolution block for light field images.
    Does convolution depthwise across the SAIs
    #use: X = LF_conv_block(X, n_sais=n_sais, img_shape=input_shape)
    '''
    n_ang = int(np.sqrt(n_sais))
    fmaps = [] # feature maps
    X = inputs
    if len(X.shape) == 6:
        X1 = X[0,:,:,:,:]
    else:
        X1 = X
    X1 = tf.reshape(X1, (img_shape[0] * img_shape[2], img_shape[1], img_shape[3], img_shape[-1]))
    X1 = layers.DepthwiseConv2D(filter_size, strides=1, padding='same', input_shape=X.shape[2:], activation='relu')(X1) 
    X1 = tf.reshape(X1, (n_ang, img_shape[1], n_ang, img_shape[3], img_shape[-1]))
    fmaps.append(X1)
    X = tf.squeeze(tf.stack(fmaps, axis=1))
    X = layers.Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(X)
    return X


class LFSEBlock(tf.keras.Model):
    '''
    Light field squeeze-and-excitation block.
    Comes after convolution and pooling blocks
    '''
    def __init__(self, n_filters, filter_size):
        super(LFSEBlock, self).__init__(name='light_field_se_block')
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.W1 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters, n_filters//4)), 
                                dtype=tf.float32), trainable=True)
        self.W2 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters//4, n_filters)), 
                                dtype=tf.float32), trainable=True)
    
    def call(self, input_tensor, training=True):
        shape = tf.shape(input_tensor) 
        height = shape[1]
        width = shape[3]
        #C = self.n_filters
        #W1 = tf.cast(tf.convert_to_tensor(np.random.randn(C, C//r)), dtype=tf.float32) 
        #W2 = tf.cast(tf.convert_to_tensor(np.random.randn(C//r, C)), dtype=tf.float32)
        W1 = self.W1
        W2 = self.W2
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




class LFSEBlock(tf.keras.Model):
    '''
    Light field squeeze-and-excitation block.
    Comes after convolution and pooling blocks
    '''
    def __init__(self, n_filters, filter_size):
        super(LFSEBlock, self).__init__(name='light_field_se_block')
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.W1 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters, n_filters//4)), 
                                dtype=tf.float32), trainable=True)
        self.W2 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters//4, n_filters)), 
                                dtype=tf.float32), trainable=True)
    
    def call(self, input_tensor, training=True):
        shape = tf.shape(input_tensor) 
        height = shape[1]
        width = shape[3]
        #C = self.n_filters
        #W1 = tf.cast(tf.convert_to_tensor(np.random.randn(C, C//r)), dtype=tf.float32) 
        #W2 = tf.cast(tf.convert_to_tensor(np.random.randn(C//r, C)), dtype=tf.float32)
        W1 = self.W1
        W2 = self.W2
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
class LFSEBlock(tf.keras.Model):
    '''
    Light field squeeze-and-excitation block.
    Comes after convolution and pooling blocks
    '''
    def __init__(self, n_filters, filter_size):
        super(LFSEBlock, self).__init__(name='light_field_se_block')
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.W1 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters, n_filters//4)), 
                                dtype=tf.float32), trainable=True)
        self.W2 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters//4, n_filters)), 
                                dtype=tf.float32), trainable=True)
    
    def call(self, input_tensor, training=True):
        shape = tf.shape(input_tensor) 
        height = shape[1]
        width = shape[3]
        #C = self.n_filters
        #W1 = tf.cast(tf.convert_to_tensor(np.random.randn(C, C//r)), dtype=tf.float32) 
        #W2 = tf.cast(tf.convert_to_tensor(np.random.randn(C//r, C)), dtype=tf.float32)
        W1 = self.W1
        W2 = self.W2
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
class LFSEBlock(tf.keras.Model):
    '''
    Light field squeeze-and-excitation block.
    Comes after convolution and pooling blocks
    '''
    def __init__(self, n_filters, filter_size):
        super(LFSEBlock, self).__init__(name='light_field_se_block')
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.W1 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters, n_filters//4)), 
                                dtype=tf.float32), trainable=True)
        self.W2 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters//4, n_filters)), 
                                dtype=tf.float32), trainable=True)
    
    def call(self, input_tensor, training=True):
        shape = tf.shape(input_tensor) 
        height = shape[1]
        width = shape[3]
        #C = self.n_filters
        #W1 = tf.cast(tf.convert_to_tensor(np.random.randn(C, C//r)), dtype=tf.float32) 
        #W2 = tf.cast(tf.convert_to_tensor(np.random.randn(C//r, C)), dtype=tf.float32)
        W1 = self.W1
        W2 = self.W2
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
class LFSEBlock(tf.keras.Model):
    '''
    Light field squeeze-and-excitation block.
    Comes after convolution and pooling blocks
    '''
    def __init__(self, n_filters, filter_size):
        super(LFSEBlock, self).__init__(name='light_field_se_block')
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.W1 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters, n_filters//4)), 
                                dtype=tf.float32), trainable=True)
        self.W2 = tf.Variable(tf.cast(tf.convert_to_tensor(np.random.randn(n_filters//4, n_filters)), 
                                dtype=tf.float32), trainable=True)
    
    def call(self, input_tensor, training=True):
        shape = tf.shape(input_tensor) 
        height = shape[1]
        width = shape[3]
        #C = self.n_filters
        #W1 = tf.cast(tf.convert_to_tensor(np.random.randn(C, C//r)), dtype=tf.float32) 
        #W2 = tf.cast(tf.convert_to_tensor(np.random.randn(C//r, C)), dtype=tf.float32)
        W1 = self.W1
        W2 = self.W2
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
