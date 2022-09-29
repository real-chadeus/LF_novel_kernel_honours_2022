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
                size_weight = tf.math.reduce_sum(curr_map)  
                mask_op = mask_op.write(i, size_weight)
            masks = masks.write(b, mask_op.stack())
        masks = masks.stack()
        masks = tf.ensure_shape(masks, shape=[self.batch_size, self.n_filters])
        return masks

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
        #s_mask = self.relative_size(f_maps)  
        results = tf.TensorArray(tf.float32, size=self.batch_size, dynamic_size=True)
        for b in range(self.batch_size):
            result = tf.TensorArray(tf.float32, size=self.n_filters, dynamic_size=True)
            for i in range(self.n_filters):
                #s_feat = lfi[b, :, :, :, :] + s_mask[b, i]
                h_feat = tf.transpose(lfi[b, :, :, :, :], perm=[0,3,2,1]) + h_mask[b, i, :]
                h_feat = tf.transpose(h_feat, perm=[0,3,2,1])
                result = result.write(i, h_feat)
                #combined_feat = s_feat + h_feat
                #result = result.write(i, combined_feat)
            results = results.write(b, result.stack())
        
        results = results.stack()
        results = tf.ensure_shape(results, 
                                 shape=[self.batch_size,
                                        self.n_filters,
                                        ] + lfi.shape[1:])
        results = tf.transpose(results, perm=[0,2,3,4,5,1])
        # aggregate over angular dimensions
        results = tf.math.reduce_sum(results, axis=[1, -2])
        #results = tf.reshape(results, [self.batch_size, 81, results.shape[3], results.shape[4], self.n_filters])
        return results


def aggregate(cost_volume):
    # aggregate cost volume
    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(cost_volume)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)
    
    return X
    

def combine(multi_view, depth_cues):
    # combine monocular depth cues with multi-view features 
    X = depth_cues + multi_view 

    X = layers.Conv2D(filters=160, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=160, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=160, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=160, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=81, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=17, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    return X


def feature_extractor(X, n_sais=81, monocular=False):

    if monocular:
        # depth cue extraction
        X = layers.LayerNormalization()(X)

        X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.LayerNormalization()(X)

        X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.LayerNormalization()(X)

        X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.LayerNormalization()(X)

        X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.LayerNormalization()(X)

        X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.LayerNormalization()(X)

        out = X

    else:
        # lfi feature extraction and cost volume creation 
        X1 = layers.Conv3D(filters=162, kernel_size=(1,1,1), padding='same')(X)
        X1 = layers.AveragePooling3D(pool_size=(9,2,2), padding='same')(X1)
        X1 = layers.LeakyReLU()(X1)
        X1 = layers.UpSampling3D(size=(1,2,2))(X1)

        X2 = layers.Conv3D(filters=162, kernel_size=(1,1,1), padding='same')(X)
        X2 = layers.AveragePooling3D(pool_size=(9,4,4), padding='same')(X2)
        X2 = layers.LeakyReLU()(X2)
        X2 = layers.UpSampling3D(size=(1,4,4))(X2)

        X3 = layers.Conv3D(filters=162, kernel_size=(1,1,1), padding='same')(X)
        X3 = layers.AveragePooling3D(pool_size=(9,8,8), padding='same')(X3)
        X3 = layers.LeakyReLU()(X3)
        X3 = layers.UpSampling3D(size=(1,8,8))(X3)

        X4 = layers.Conv3D(filters=162, kernel_size=(1,1,1), padding='same')(X)
        X4 = layers.AveragePooling3D(pool_size=(9,16,16), padding='same')(X4)
        X4 = layers.LeakyReLU()(X4)
        X4 = layers.UpSampling3D(size=(1,16,16))(X4)

        output_features = layers.Concatenate(axis=1)([X1, X2, X3, X4])
        out = tf.reduce_sum(output_features, axis=1)
        out = layers.Conv2D(filters=162, kernel_size=(1,1), padding='same', activation='relu')(out)

    return out

def disp_regression(X):
    shape = X.shape
    disparity_values = np.linspace(-4, 4, 17)
    x = tf.constant(disparity_values, shape=[17])
    x = tf.expand_dims(tf.expand_dims(tf.expand_dims(x, 0), 0), 0)
    x = tf.tile(x, (shape[0], shape[1], shape[2], 1))
    out = multiply([X,x])
    out = tf.squeeze(out)
    out = tf.math.reduce_sum(out, axis=-1)
    return out

def build_model(input_shape, summary=True, n_sais=81, angres=9, batch_size=16):
    '''
    build the model
    param output_shape: size of the 2D depth map
    '''
    # initial input mapping
    inputs = keras.Input(shape=input_shape, name='model_input', batch_size=batch_size)
    X = inputs
    
    # monocular feature depth cue from center view
    center = angres//2 
    center_view = X[:,center,:,:,center]
    center_view = tf.expand_dims(center_view, axis=-1)
    f_maps = feature_extractor(center_view, n_sais=n_sais, monocular=True)
    depth_cues = DepthCueExtractor(h=X.shape[2], w=X.shape[3], 
                        n_filters=162, batch_size=batch_size)(X, f_maps)

    # multi-view feature extraction + cost volume creation
    X = feature_extractor(X, n_sais=n_sais)
    #X = aggregate(X)
    # integrate cost volume & depth cues
    X = combine(X, depth_cues)

    #X = generate_cost(X, n_filters=170) 
    predictions = disp_regression(X)

    model = models.Model(inputs=inputs, outputs=predictions) 
 
    if summary:
        model.summary()

    return model












