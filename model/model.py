from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.models as models
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import einops
import matplotlib.pyplot as plt
from model.conv4d import Conv4D

class DepthCueExtractor(tf.keras.Model):
    '''
    extracts monocular depth cue information
    '''
    def __init__(self, h, w, angres, n_filters, batch_size):
        super(DepthCueExtractor, self).__init__(name='depth_cue_extractor')
        self.h = h
        self.w = w
        self.angres = angres
        self.n_filters = n_filters
        self.size_mask = tf.ones((batch_size, n_filters)) 
        self.height_mask = tf.ones((batch_size, n_filters, h)) 
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
                temp = self.size_mask[b, i] * size_weight
                mask_op.write(i, temp)
            masks.write(b, mask_op.stack())
        self.size_mask = masks.stack()

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
                height_vectors = tf.math.reduce_mean(curr_map, axis=0)  
                height_vectors = height_vectors/tf.math.reduce_max(height_vectors)
                temp = self.height_mask[b, i, :] * height_vectors
                mask_op.write(i, temp)
            masks.write(b, mask_op.stack())
        return masks.stack() 

    def call(self, lfi, f_maps):
        h_mask = self.height(f_maps)
        #self.relative_size(f_maps)  
        results = tf.TensorArray(tf.float32, size=self.batch_size, dynamic_size=True)
        for b in range(self.batch_size):
            result = tf.TensorArray(tf.float32, size=self.batch_size, dynamic_size=True)
            for i in range(self.n_filters):
                #s_feat = lfi[b, :, :, :, :] * self.size_mask[b, i]
                h_feat = tf.transpose(lfi[b, :, :, :, :], perm=[0,3,2,1]) * h_mask[b, i, :]
                h_feat = tf.transpose(h_feat, perm=[0,3,2,1])
                result.write(i, h_feat)
            results.write(b, result.stack())
        
        results = results.stack()
        # aggregate across channels
        results = tf.ensure_shape(results, 
                                 shape=[self.batch_size,
                                        self.n_filters,
                                        ] + lfi.shape[1:])
        results = tf.reduce_mean(results, axis=5)
        results = tf.transpose(results, perm=[0,2,3,4,1])
        return results

class OcclusionHandler(tf.keras.Model):
    '''
    generates occlusion aware matching costs.
    '''
    def __init__(self, h, w, angres, f_maps):
        super(OcclusionHandler, self).__init__(name='occlusion_handler')
        self.angres = angres
        self.h = h
        self.w = w
        # occlusion mask is trainable; initialized with values of 1
        self.mask = tf.ones((angres, h, angres, w, 1)) 
        self.cost = self.build_cost(f_maps, self.mask)
        self.disp = self.aggregate(self.cost)

    def warp(img, disp, du, dv, x_base, y_base):
        return None
    
    def occlusion_mask(self, lfi, disp):
        # create mask to handle occlusions
        return None

    def build_cost(self, x, mask):
        # apply mask to the image
        cost = x * mask
        return cost

    def call(self, img, f_maps):
        self.mask = self.occlusion_mask(img, self.disp)
        self.cost = self.build_cost(f_maps, mask)
        self.disp = self.aggregate(cost)
        return self.disp

def aggregate(cost_volume, depth_cues=None, combine=False, n_filters=24):
    # aggregate the cost volume by combining features (incl. monocular depth cues)
    X = layers.Conv3D(filters=1, kernel_size=(1,1,1), padding='same')(cost_volume)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=n_filters//2, kernel_size=(1,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=n_filters//2, kernel_size=(1,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=n_filters//2, kernel_size=(1,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=n_filters, kernel_size=(1,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=n_filters, kernel_size=(1,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    
    if combine:
        # combine monocular depth cues with multi-view features 
        depth_cues = layers.AveragePooling3D(pool_size=(1,2,2))(depth_cues)
        depth_cues = layers.LeakyReLU()(depth_cues)
        depth_cues = layers.BatchNormalization()(depth_cues)
        depth_cues = layers.AveragePooling3D(pool_size=(1,2,2))(depth_cues)
        X = depth_cues * X 
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)

    X = layers.AveragePooling3D(pool_size=(9,1,1))(X)
    X = layers.LeakyReLU()(X)
    X = layers.BatchNormalization()(X)
    X = layers.AveragePooling3D(pool_size=(9,1,1))(X)
    X = layers.LeakyReLU()(X)
    X = layers.BatchNormalization()(X)
    X = tf.squeeze(X)

    return X


def feature_extractor(X, n_sais=81, monocular=False, input_shape=(436,436,3)):

    if monocular:
        # depth cue extraction
        X = layers.Conv2D(filters=1, kernel_size=(1,1), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=2, kernel_size=(9,9), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=2, kernel_size=(9,9), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=3, kernel_size=(16,16), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=3, kernel_size=(16,16), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=9, kernel_size=(9,9), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=9, kernel_size=(9,9), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=24, kernel_size=(4,4), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=24, kernel_size=(4,4), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=60, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=60, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)

    else:
        # lfi feature extraction
        X = layers.Conv3D(filters=1, kernel_size=(1,9,9), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=2, kernel_size=(1,9,9), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.AveragePooling3D(pool_size=(1,2,2))(X)
        X = layers.Conv3D(filters=3, kernel_size=(1,6,6), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=3, kernel_size=(1,6,6), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.AveragePooling3D(pool_size=(1,2,2))(X)
        X = layers.Conv3D(filters=6, kernel_size=(1,6,6), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=6, kernel_size=(1,3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=9, kernel_size=(1,3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=9, kernel_size=(1,3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)

    return X


def build_model(input_shape, summary=True, n_sais=81, angres=9, batch_size=16):
    '''
    build the model
    param output_shape: size of the 2D depth map
    '''
    # initial input mapping
    inputs = keras.Input(shape=input_shape, name='lfse_model_input', batch_size=batch_size)
    X = layers.LeakyReLU()(inputs)
    X = layers.BatchNormalization()(X)
    
    # monocular feature extraction from center view
    center = n_sais//2 
    center_view = X[:,center,:,:,:]
    f_maps = feature_extractor(center_view, n_sais=n_sais, monocular=True)
    # monocular depth cue extraction
    depth_cues = DepthCueExtractor(h=X.shape[2], w=X.shape[3], 
                        angres=int(np.sqrt(input_shape[1])), n_filters=60,
                                    batch_size=batch_size)(X, f_maps)

    # multi-view feature extraction across the entire lfi
    X = feature_extractor(X, n_sais=n_sais)
    # occlusion masking 
    #X = OcclusionHandler()(X)

    # aggregate multi-view and monocular features
    X = aggregate(X, depth_cues=depth_cues, combine=True, n_filters=60)

    X = layers.Dense(2048, activation='linear')(X)
    X = layers.LeakyReLU()(X)
    X = layers.Dense(4096, activation='sigmoid')(X)
    X = layers.Dense(2048, activation='linear')(X)
    X = layers.LeakyReLU()(X)
    X = layers.Dense(4096, activation='sigmoid')(X)
    #X = layers.BatchNormalization()(X)

    X = layers.Conv2DTranspose(filters=1, strides=4, kernel_size=(3,3), padding='same')(X)

    X = tf.squeeze(layers.Dense(1, activation='linear')(X))

    model = models.Model(inputs=inputs, outputs=X)   
 
    if summary:
        model.summary()

    return model











