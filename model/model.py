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
                height_vectors = tf.math.reduce_mean(curr_map, axis=0)  
                height_vectors = height_vectors/tf.math.reduce_max(height_vectors)
                mask_op = mask_op.write(i, height_vectors)
            masks = masks.write(b, mask_op.stack())
        masks = masks.stack()
        masks = tf.ensure_shape(masks, 
                                shape=[self.batch_size, 
                                    self.n_filters, self.h])
        return masks 

    def call(self, lfi, f_maps):
        h_mask = self.height(f_maps)
        s_mask = self.relative_size(f_maps)  
        results = tf.TensorArray(tf.float32, size=self.batch_size, dynamic_size=True)
        for b in range(self.batch_size):
            result = tf.TensorArray(tf.float32, size=self.n_filters, dynamic_size=True)
            for i in range(self.n_filters):
                s_feat = lfi[b, :, :, :, :] * s_mask[b, i]
                h_feat = tf.transpose(lfi[b, :, :, :, :], perm=[0,3,2,1]) * h_mask[b, i, :]
                h_feat = tf.transpose(h_feat, perm=[0,3,2,1])
                #result = result.write(i, h_feat)
                combined_feat = s_feat * h_feat
                result = result.write(i, combined_feat)
            results = results.write(b, result.stack())
        
        results = results.stack()
        results = tf.ensure_shape(results, 
                                 shape=[self.batch_size,
                                        self.n_filters,
                                        ] + lfi.shape[1:])
        results = tf.transpose(results, perm=[0,2,3,4,5,1])
        results = tf.reduce_mean(results, axis=(1,4))
        return results


def aggregate(cost_volume):
    # aggregate cost volume
    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(cost_volume)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=162, kernel_size=(3,3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    
    return X
    

def combine(cost, depth_cues):
    # combine monocular depth cues with multi-view features 
    X = depth_cues * cost 
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=170, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)
    X = layers.Conv2D(filters=170, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=170, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = layers.Conv2D(filters=170, kernel_size=(3,3), padding='same')(X)
    X = layers.LeakyReLU()(X)
    X = layers.LayerNormalization()(X)

    X = tf.squeeze(X)

    return X

def generate_cost(f_maps, n_filters=24):
    map_shape = f_maps[:,:,:,:,0].shape
    disparity_values = np.linspace(-4, 4, 17) # limit disparity range to -4, 4
    disparity_costs = []
    for d in disparity_values:
        if d == 0:
            tmp_list = []
            for i in range(n_filters):
                tmp_list.append(f_maps[:,:,:,:,i])
        else:
            tmp_list = []
            for i in range(n_filters):
                (v, u) = divmod(i, 9)
                tensor = tf.roll(f_maps[:,:,:,:,i],
                                             [int(d * (u - 4)), int(d * (v - 4))],
                                             axis=[2,3])
                tmp_list.append(tensor)

        cost = tf.concat(tmp_list, axis=1)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1)
    cost_volume = tf.reshape(cost_volume,
                            (map_shape[0], 17, map_shape[2], map_shape[3], n_filters))
    cost_volume = layers.AveragePooling3D(pool_size=(3,1,1))(cost_volume)
    cost_volume = layers.AveragePooling3D(pool_size=(3,1,1))(cost_volume)
    return cost_volume

def feature_extractor(X, n_sais=81, monocular=False):

    if monocular:
        # depth cue extraction
        X = layers.LayerNormalization()(X)

        X = tf.clip_by_value(X, -4, 4)
        X = layers.Conv2D(filters=162, kernel_size=(1,1), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.LayerNormalization()(X)

        X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.LayerNormalization()(X)

        X = layers.Conv2D(filters=162, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.LayerNormalization()(X)

        X = layers.Conv2D(filters=243, kernel_size=(3,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.LayerNormalization()(X)


    else:
        # lfi feature extraction and cost volume creation 
        X = tf.clip_by_value(X, -4, 4)
        X = layers.Conv3D(filters=162, kernel_size=(1,3,3), padding='same')(X)

        X = layers.Conv3D(filters=162, kernel_size=(1,3,3), padding='same')(X)
        X = layers.AveragePooling3D(pool_size=(3,1,1))(X)

        X = layers.Conv3D(filters=162, kernel_size=(1,3,3), padding='same')(X)
        X = layers.AveragePooling3D(pool_size=(3,1,1))(X)

        X = layers.Conv3D(filters=162, kernel_size=(1,3,3), padding='same')(X)
        X = layers.AveragePooling3D(pool_size=(1,1,1))(X)

        X = layers.Conv3D(filters=162, kernel_size=(1,3,3), padding='same')(X)
        X = layers.LayerNormalization()(X)
        
        #X = generate_cost(X, n_filters=81) 

    return X


def build_model(input_shape, summary=True, n_sais=81, angres=9, batch_size=16):
    '''
    build the model
    param output_shape: size of the 2D depth map
    '''
    # initial input mapping
    inputs = keras.Input(shape=input_shape, name='model_input', batch_size=batch_size)
    X = layers.BatchNormalization()(inputs)
    
    # monocular feature depth cue from center view
    center = angres//2 
    center_view = X[:,center,:,:,center]
    center_view = tf.expand_dims(center_view, axis=-1)
    f_maps = feature_extractor(center_view, n_sais=n_sais, monocular=True)
    depth_cues = DepthCueExtractor(h=X.shape[2], w=X.shape[3], 
                        n_filters=162, batch_size=batch_size)(X, f_maps)

    # multi-view feature extraction + cost volume creation
    X = feature_extractor(X, n_sais=n_sais)
    # aggregate cost volume
    X = aggregate(X)
    # integrate cost volume & depth cues
    X = combine(X, depth_cues)

    if batch_size > 1:
        X = tf.reduce_mean(X, axis=1)

    predictions = tf.squeeze(layers.Dense(1, activation='linear')(X))
    predictions = predictions * 0.1

    model = models.Model(inputs=inputs, outputs=predictions) 
 
    if summary:
        model.summary()

    return model












