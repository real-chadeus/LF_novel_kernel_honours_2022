from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.models as models
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import einops
import matplotlib.pyplot as plt
from model.conv4d import Conv4D
import tensorflow_addons as tfa
from tensorflow.keras.layers import multiply
import sys


class DepthCueExtractor(tf.keras.Model):
    '''
    extracts monocular depth cue information
    '''
    def __init__(self, h, w, n_filters, batch_size):
        super(DepthCueExtractor, self).__init__(name='depth_cue_extractor')
        self.h = h
        self.w = w
        self.n_filters = n_filters
        self.batch_size = batch_size

    @tf.function
    def relative_size(self, f_maps):
        '''
        extracts relative size through center view features
        gets the reduce sum of the pixel values of the current feature map
        '''
        masks = tf.TensorArray(tf.float32, size=self.batch_size, dynamic_size=True)
        for b in range(self.batch_size):
            mask_op = tf.TensorArray(tf.float32, size=self.n_filters, dynamic_size=True)
            for i in range(self.n_filters):
                curr_map = f_maps[b, :, :, i]
                curr_map = tf.squeeze(curr_map)
                size_weight = tf.math.reduce_euclidean_norm(curr_map)  
                mask_op = mask_op.write(i, size_weight)
            masks = masks.write(b, mask_op.stack())
        masks = masks.stack()
        masks = tf.ensure_shape(masks, shape=[self.batch_size, self.n_filters])
        self.s_weight = self.s_weight + masks

    @tf.function
    def height(self, f_maps):
        '''
        extracts height in plane from the feature map
        '''
        masks = tf.TensorArray(tf.float32, size=self.batch_size, dynamic_size=True)
        for b in range(self.batch_size):
            mask_op = tf.TensorArray(tf.float32, size=self.n_filters, dynamic_size=True)
            for i in range(self.n_filters):
                curr_map = f_maps[b, :, :, i]
                curr_map = tf.squeeze(curr_map)
                height_vectors = tf.math.reduce_sum(curr_map, axis=0) 
                mask_op = mask_op.write(i, height_vectors)
            masks = masks.write(b, mask_op.stack())
        masks = masks.stack()
        masks = tf.ensure_shape(masks, 
                                shape=[self.batch_size, 
                                    self.n_filters, self.h])
        return masks 
    
    def call(self, lfi, f_maps):
        h_mask = self.height(f_maps)
        #self.relative_size(f_maps)  
        results = tf.TensorArray(tf.float32, size=self.batch_size, dynamic_size=True)
        #tf.print(lfi, output_stream = tf.compat.v1.logging.info, summarize=-1)
        for b in range(self.batch_size):
            result = tf.TensorArray(tf.float32, size=self.n_filters, dynamic_size=True)
            for i in range(self.n_filters):
                #s_feat = lfi[b, :, :, :, :] + self.s_weight[b, i]
                #tf.print(self.h_mask, output_stream = tf.compat.v1.logging.info, summarize=-1)
                h_feat = tf.transpose(lfi[b, :, :, :, :], perm=[0,3,2,1]) * h_mask[b, i, :]
                h_feat = tf.transpose(h_feat, perm=[0,3,2,1])
                result = result.write(i, h_feat)
                #combined_feat = s_feat * h_feat
                #result = result.write(i, combined_feat)
            results = results.write(b, result.stack())
        
        results = results.stack()
        results = tf.ensure_shape(results, 
                                 shape=[self.batch_size,
                                        self.n_filters,
                                        ] + lfi.shape[1:])
        results = tf.transpose(results, perm=[0,2,3,4,5,1])
        # aggregate over angular dimensions
        results = tf.math.reduce_mean(results, axis=[1,4])
        #tf.print(results, output_stream = tf.compat.v1.logging.info, summarize=-1)
        return results


class Tester(tf.keras.Model):
    '''
    extracts monocular depth cue information
    '''
    def __init__(self):
        super(Tester, self).__init__(name='tester')

    def call(self, X):
        tf.print(X, output_stream = sys.stdout, summarize=-1)
        return X

def aggregate(cost_volume):
    # aggregate cost volume
    X = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same')(cost_volume)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=512, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=81, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=12, kernel_size=(3,3), padding='same')(X)

    return X
    

def combine(multi_view, depth_cues):
    # combine monocular depth cues with multi-view features 
    X = multi_view + depth_cues 
    X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)

    X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=81, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=24, kernel_size=(3,3), padding='same')(X)

    return X


def feature_extractor(X, monocular=False):
    # lfi feature extraction and cost volume creation 
    X = layers.Conv3D(filters=162, kernel_size=(1,1,1), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv3D(filters=162, kernel_size=(1,1,1), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv3D(filters=162, kernel_size=(1,1,1), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv3D(filters=162, kernel_size=(1,1,1), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    out = tf.math.reduce_mean(X, axis=1)

    return out

def monocular_extractor(X):
    # feature extraction for center view
    X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=81, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=12, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    return X

def disp_regression(X):
    shape = X.shape
    disparity_values = np.linspace(-4, 4, 24)
    x = tf.constant(disparity_values, shape=[24])
    x = tf.expand_dims(tf.expand_dims(tf.expand_dims(x, 0), 0), 0)
    x = tf.tile(x, (shape[0], shape[1], shape[2], 1))
    out = multiply([X,x])
    out = tf.math.reduce_sum(out, axis=-1)
    return out

def build_model(input_shape, summary=True, n_sais=81, angres=9, batch_size=16):
    '''
    build the model
    param output_shape: size of the 2D depth map
    '''
    # initial input mapping
    inputs = layers.Input(shape=input_shape, name='model_input', batch_size=batch_size)
    X = inputs

    # monocular depth cues
    center = angres//2 
    center_view = X[:,center,:,:,center]
    center_view = tf.expand_dims(center_view, axis=-1)
    f_maps = monocular_extractor(center_view)
    #s_weight = tf.Variable(tf.zeros(shape=(batch_size, 162)))
    depth_cues = DepthCueExtractor(h=X.shape[2], w=X.shape[3], 
              n_filters=12, batch_size=batch_size)(lfi=X, f_maps=f_maps)

    # multi-view feature extraction + cost volume creation
    X = feature_extractor(X)
    X = aggregate(X)
    # integrate cost volume & depth cues
    X = combine(X, depth_cues)
    X = layers.Activation('softmax')(X)

    #predictions = disp_regression(X)
    predictions = disp_regression(X)
    #predictions = tf.clip_by_value(predictions, 0, 4)
    #predictions = Tester()(predictions)

    model = models.Model(inputs=inputs, outputs=predictions) 
 
    if summary:
        model.summary()

    return model




