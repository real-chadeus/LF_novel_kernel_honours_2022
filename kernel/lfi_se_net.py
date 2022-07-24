from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

def LF_conv_block(inputs, n_filters=4, filter_size=(3,3), n_sais=49, stride=2): 
    # convolution block for light field images
    # layer normalization done separately
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
    # build the model using Tensorflow.
    X = layers.Conv3D(filters=3, kernel_size=(3,3,3), strides=(1,2,2), padding='same')(inputs) 
    X = tf.nn.relu(x)
    
    X = LF_conv_block(X, n_filters=3, filter_size=(3,3)
    X = layers.MaxPooling3D(pool_size=(1,2,2))

class LFSEBlock(tf.keras.Model):
    # light field squeeze-and-excitation block
    # comes after all the convolution blocks
    def __init__(self, kernel_size, filters):
        super(LFSEBlock).__init__(name='light_field_se_block')
    
    def call(self, input_tensor, training=True):
        return tf.nn.linear(X)

def build_model(shape):
    inputs = layers.Input(shape=shape, name='lfi_input')  











