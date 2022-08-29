from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.models as models
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import einops

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

class DepthCueExtractor(tf.keras.Model):
    '''
    extracts monocular depth cue information
    '''
    def __init__(self, h, w, angres):
        super(CostConstructor, self).__init__(name='depth_cue_extractor')
    
    def relative_size(self):
        return None

    def height_in_plane(self):
        return None
    
    def call(self):
        return None

class CostConstructor(tf.keras.Model):
    '''
    generates occlusion aware matching costs.
    '''
    def __init__(self, h, w, angres, f_maps):
        super(CostConstructor, self).__init__(name='cost_constructor')
        self.angres = angres
        self.h = h
        self.w = w
        # occlusion mask is trainable; initialized with values of 1
        self.mask = tf.Variable(tf.cast(tf.convert_to_tensor(tf.ones((angres, h, angres, w, 1))), 
                                dtype=tf.float32), trainable=True)
        
        self.cost = self.build_cost(f_maps, self.mask)
        self.disp = self.aggregate(self.cost)

    def aggregate(self, cost_volume, n_filters=24):
        # aggregate the cost volume by extracting relevant features
        X = layers.Conv3D(n_filters, kernel_size=(1,1,1), padding='same')(cost_volume)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        X = layers.Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        X = layers.Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        X = layers.Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        X = layers.Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
        X = layers.Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(X)
        X = layers.BatchNormalization()(X)
        X = layers.LeakyReLU()(X)
    
    def warp(self, img, disp, du, dv, x_base, y_base):
        # du and dv -> difference in angular position between current view and center view
        h = img.shape[1]
        w = img.shape[3]
        x_shifts = dv * disp[:, 0, :, :] / w
        y_shifts = du * disp[:, 0, :, :] / h
        flow_field = tf.stack((x_base + x_shifts, y_base + y_shifts), axis=1)
        img_warped = 1 
        return img_warped
    
    def occlusion_mask(self, lfi, disp):
        # create mask to handle occlusions
        angres = lfi.shape[0] 
        h = lfi.shape[1]
        w = lfi.shape[3]
        x_base = tf.repeat(tf.linspace(0,1,w), [1,h,1]) 
        y_base = tf.transpose(tf.repeat(tf.linspace(0,1,h), [1,w,1]))
        center = (angres - 1)//2 # center view
        img_ref = lfi[center, :, center, :, :]
        result = []
        for u in range(angres):
            for v in range(angres):
                img = lfi[u, :, v, :, :]
                if u == center and v == center:
                    img_warped = img
                else:
                    du, dv = u - center, v - center
                    img_warped = warp(img, -disp, du, dv, x_base, y_base)
                result.append(abs((img_warped - img_ref)))
        mask = tf.concat(result, axis=1)
        return mask

    def build_cost(self, x, mask):
        # apply mask to the image
        cost = x * mask
        return cost

    def call(self, img, f_maps):
        self.mask = self.occlusion_mask(img, self.disp)
        self.cost = self.build_cost(f_maps, mask)
        self.disp = self.aggregate(cost)
        return self.disp


def LF_conv_block(inputs, n_filters=4, 
                    filter_size=(3,3), n_sais=49, 
                    stride=2, img_shape=(7,512,7,512,3),
                    n_lfi=1): 
    '''
    Simple convolution block for light field images.
    Does convolution depthwise across the SAIs
    '''
    n_ang = int(np.sqrt(n_sais))
    fmaps = [] # feature maps
    X = inputs
    for i in range(n_lfi):
        if len(X.shape) == 6:
            X1 = X[i,:,:,:,:]
        else:
            X1 = X
        X1 = tf.reshape(X1, (img_shape[0] * img_shape[2], img_shape[1], img_shape[3], img_shape[-1]))
        X1 = layers.DepthwiseConv2D(filter_size, strides=1, padding='same', input_shape=X.shape[2:], activation='relu')(X1) 
        X1 = tf.reshape(X1, (n_ang, img_shape[1], n_ang, img_shape[3], img_shape[-1]))
        fmaps.append(X1)
    X = tf.squeeze(tf.stack(fmaps, axis=1))
    X = layers.Conv3D(n_filters, kernel_size=(3,3,3), padding='same')(X)
    return X

def build_model(input_shape, summary=True, n_sais=49):
    '''
    build the model
    param output_shape: size of the 2D depth map
    '''

    # initial input and convolution + layer normalization
    inputs = keras.Input(shape=input_shape, name='lfse_model_input')
    X = layers.Conv3D(filters=3, kernel_size=(3,3,3), padding='same')(inputs) 
    
    X = LF_conv_block(X, n_filters=3, filter_size=(3,3),img_shape=input_shape, n_sais=n_sais)

    X = layers.MaxPooling3D(pool_size=(2,1,2))(X)
    X = LF_conv_block(X, n_filters=6, filter_size=(3,3), img_shape=X.shape, n_sais=n_sais) 
    X = LF_conv_block(X, n_filters=6, filter_size=(3,3), img_shape=X.shape, n_sais=n_sais)
    X = layers.LeakyReLU()(X)
    X = layers.BatchNormalization()(X)

    X = layers.MaxPooling3D(pool_size=(2,1,2))(X)
    X = LF_conv_block(X, n_filters=12, filter_size=(3,3), img_shape=X.shape, n_sais=n_sais) 
    X = LF_conv_block(X, n_filters=12, filter_size=(3,3), img_shape=X.shape, n_sais=n_sais)
    X = layers.LeakyReLU()(X)
    X = layers.BatchNormalization()(X)

    X = LF_conv_block(X, n_filters=24, filter_size=(3,3), img_shape=X.shape, n_sais=n_sais) 
    X = LF_conv_block(X, n_filters=24, filter_size=(3,3), img_shape=X.shape, n_sais=n_sais)
    X = layers.LeakyReLU()(X)
    X = layers.BatchNormalization()(X)
    
    X = CostConstructor(h=X.shape[1], w=X.shape[3], 
                        angres = input_shape[0], f_maps=X)(img=inputs, f_maps=X)

    X = LFSEBlock(n_filters=24, filter_size=(3,3))(X)
    X = layers.BatchNormalization()(X)
    
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












