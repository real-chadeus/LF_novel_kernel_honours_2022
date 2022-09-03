from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.models as models
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import einops
import matplotlib.pyplot as plt

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

class DepthCueExtractor(tf.keras.Model):
    '''
    extracts monocular depth cue information
    '''
    def __init__(self, h, w, angres, n_filters):
        super(DepthCueExtractor, self).__init__(name='depth_cue_extractor')
        self.h = h
        self.w = w
        self.angres = angres
        self.n_filters = n_filters
        self.size_mask = tf.ones((n_filters, w)) 
        self.height_mask = tf.ones((n_filters, w)) 

    def relative_size(self, f_maps, center_view):
        '''
        extracts relative size through center view features
        '''
        for i in range(self.n_filters):
            curr_map = f_maps[:, :, :, i]
            curr_map = tf.squeeze(curr_map)
            size_vectors = tf.math.reduce_sum(curr_map, axis=0)  
            size_vectors = size_vectors/tf.math.reduce_max(size_vectors)
            self.size_mask[i] *= size_vectors
        return None

    def height(self, f_maps, center_view):
        '''
        extracts height in plane from the feature map
        '''
        for i in range(self.n_filters):
            curr_map = f_maps[:, :, :, i]
            curr_map = tf.squeeze(curr_map)
            height_vectors = tf.math.reduce_mean(curr_map, axis=0))  
            height_vectors = height_vectors/tf.math.reduce_max(height_vectors)
            self.height_mask[i] *= height_vectors
    
    def call(self, lfi, f_map):
        center = self.angres // 2 
        center_view = lfi[center, :, center, :, :]
        self.height(f_map, center_view)
        self.relative_size(f_map, center_view)  
        result = tf.ones(shape=(2*self.angres**2,self.h,self.w,3), 
                            dtype=tf.float32)
        for u in range(2*angres,step=2):
            for v in range(2*angres,step=2):
                view = lfi[u, :, v, :, :]
                result1 = view * self.height_mask
                result2 = view * self.size_mask
                result[u+1, :, v+1, :, :] = result1  
                result[u, :, v, :, :] = result2

        result = layers.AveragePooling3D(pool_size=(2,1,2))(result) 
        result = layers.LeakyReLU()(result)
        result = layers.BatchNormalization()(result)
        return lfi

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
    X = layers.Conv3D(filters=n_filters//2, kernel_size=(3,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=n_filters//2, kernel_size=(3,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=n_filters//2, kernel_size=(3,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=n_filters, kernel_size=(3,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    X = layers.Conv3D(filters=n_filters, kernel_size=(3,3,3), padding='same')(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU()(X)
    
    if combine:
        # combine monocular depth cues with multi-view features 
         

    X = tf.math.reduce_mean(X, axis=(0,2))
    X = layers.LeakyReLU()(X)
    X = layers.BatchNormalization()(X)
    return X


def feature_extractor(X, n_sais=81, monocular=False, input_shape=(436,436,3)):

    if monocular:
        # depth cue extraction
        X = layers.Conv2D(filters=1, kernel_size=(1,1))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=2, kernel_size=(9,9))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=2, kernel_size=(9,9))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=3, kernel_size=(16,16))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=3, kernel_size=(16,16))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=9, kernel_size=(9,9))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=9, kernel_size=(9,9))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=32, kernel_size=(4,4))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=32, kernel_size=(4,4))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=64, kernel_size=(3,3))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv2D(filters=64, kernel_size=(3,3))(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)

    else:
        # lfi feature extraction
        X = layers.Conv3D(filters=1, kernel_size=(9,1,9), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=2, kernel_size=(9,1,9), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.AveragePooling3D(pool_size=(2,1,2))(X)
        X = layers.Conv3D(filters=3, kernel_size=(6,1,6), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=3, kernel_size=(6,1,6), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.AveragePooling3D(pool_size=(2,1,2))(X)
        X = layers.Conv3D(filters=6, kernel_size=(3,1,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=6, kernel_size=(3,1,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=9, kernel_size=(3,1,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)
        X = layers.Conv3D(filters=9, kernel_size=(3,1,3), padding='same')(X)
        X = layers.LeakyReLU()(X)
        X = layers.BatchNormalization()(X)

    return X

def build_model(input_shape, summary=True, n_sais=81, angres=9):
    '''
    build the model
    param output_shape: size of the 2D depth map
    '''
    # initial input mapping
    inputs = keras.Input(shape=input_shape, name='lfse_model_input')
    X = layers.LeakyReLU()(inputs)
    X = layers.BatchNormalization()(X)
    
    X = X[0,:,:,:,:]
    # monocular feature extraction from center view
    center = angres//2 
    center_view = X[center,:,center,:]
    center_view = tf.expand_dims(center_view, axis=0)
    f_maps = feature_extractor(center_view, n_sais=n_sais, monocular=True)
    # monocular depth cue extraction
    depth_cues = DepthCueExtractor(h=X.shape[1], w=X.shape[3], 
                        angres = input_shape[0], n_filters=64)(X, f_maps)

    # multi-view feature extraction across the entire lfi
    X = feature_extractor(X, n_sais=n_sais)
    # occlusion masking 
    #X = OcclusionHandler()(X)

    # aggregate multi-view and monocular features
    X = aggregate(X, depth_cues=depth_cues, combine=True)

    X = layers.Dense(2048, activation='linear')(X)
    X = layers.LeakyReLU()(X)
    X = layers.Dense(4096, activation='sigmoid')(X)
    X = layers.Dense(2048, activation='linear')(X)
    X = layers.LeakyReLU()(X)
    X = layers.Dense(4096, activation='sigmoid')(X)

    X = tf.expand_dims(X, axis=0)
    X = layers.Conv2DTranspose(filters=1, strides=4, kernel_size=(3,3), padding='same')(X)

    X = tf.squeeze(layers.Dense(1, activation='linear')(X))

    model = models.Model(inputs=inputs, outputs=X)   
 
    if summary:
        model.summary()

    return model












