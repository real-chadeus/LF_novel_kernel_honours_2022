from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, Reshape, Conv3D, AveragePooling2D, Lambda, UpSampling2D, UpSampling3D, GlobalAveragePooling3D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, add, multiply

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import time
import tensorflow_addons as tfa

class DepthCueExtractor(tf.keras.Model):
    '''
    extracts monocular depth cue information
    '''
    def __init__(self):
        super(DepthCueExtractor, self).__init__(name='depth_cue_extractor')

    def relative_size(self, f_maps):
        '''
        extracts relative size through center view features
        gets the reduce sum of the pixel values of the current feature map
        '''
        size_weight = tf.math.reduce_mean(f_maps)  
        s_mask = size_weight * f_maps
        return s_mask

    def height(self, f_maps):
        '''
        extracts height in plane from the feature map
        '''
        height_vectors = tf.math.reduce_mean(f_maps, axis=2) 
        h_mask = height_vectors * f_maps
        return h_mask 
    
    def call(self, f_maps):
        h_mask = self.height(f_maps)
        s_mask = self.relative_size(f_maps)
        result = h_mask * s_mask
        return result

def convbn(inputs, out_planes, kernel_size, stride, dilation):

    seq = Conv2D(out_planes,
                 kernel_size,
                 stride,
                 'same',
                 dilation_rate=dilation,
                 use_bias=False)(inputs)
    seq = BatchNormalization()(seq)
    return seq

def convbn_3d(inputs, out_planes, kernel_size, stride):
    seq = Conv3D(out_planes, kernel_size, stride, 'same',
                 use_bias=False)(inputs)
    seq = BatchNormalization()(seq)
    return seq

def BasicBlock(inputs, planes, stride, downsample, dilation):
    conv1 = convbn(inputs, planes, 3, stride, dilation)
    conv1 = Activation('relu')(conv1)
    conv2 = convbn(conv1, planes, 3, 1, dilation)
    if downsample is not None:
        inputs = downsample

    conv2 = add([conv2, inputs])
    return conv2


def _make_layer(inputs, planes, blocks, stride, dilation):
    inplanes = 4
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = Conv2D(planes, 1, stride, 'same', use_bias=False)(inputs)
        downsample = BatchNormalization()(downsample)

    layers = BasicBlock(inputs, planes, stride, downsample, dilation)
    for i in range(1, blocks):
        layers = BasicBlock(layers, planes, 1, None, dilation)

    return layers


def UpSampling2DBilinear(size):
    return Lambda(lambda x: tf.compat.v1.image.resize_bilinear(
        x, size, align_corners=True))


def UpSampling3DBilinear(size):

    def UpSampling3DBilinear_(x, size):
        shape = K.shape(x)
        x = K.reshape(x, (shape[0] * shape[1], shape[2], shape[3], shape[4]))
        x = tf.image.resize_bilinear(x, size, align_corners=True)
        x = K.reshape(x, (shape[0], shape[1], size[0], size[1], shape[4]))
        return x

    return Lambda(lambda x: UpSampling3DBilinear_(x, size))


def feature_extraction(sz_input, sz_input2):
    i = Input(shape=(sz_input, sz_input2, 1))
    firstconv = convbn(i, 4, 3, 1, 1)
    firstconv = Activation('relu')(firstconv)
    firstconv = convbn(firstconv, 4, 3, 1, 1)
    firstconv = Activation('relu')(firstconv)

    layer1 = _make_layer(firstconv, 4, 2, 1, 1)
    layer2 = _make_layer(layer1, 8, 8, 1, 1)
    layer3 = _make_layer(layer2, 16, 2, 1, 1)
    layer4 = _make_layer(layer3, 16, 2, 1, 2)

    layer4_size = (layer4.get_shape().as_list()[1],
                   layer4.get_shape().as_list()[2])

    branch1 = AveragePooling2D((2, 2), (2, 2), 'same')(layer4)
    branch1 = convbn(branch1, 4, 1, 1, 1)
    branch1 = Activation('relu')(branch1)
    branch1 = UpSampling2DBilinear(layer4_size)(branch1)

    branch2 = AveragePooling2D((4, 4), (4, 4), 'same')(layer4)
    branch2 = convbn(branch2, 4, 1, 1, 1)
    branch2 = Activation('relu')(branch2)
    branch2 = UpSampling2DBilinear(layer4_size)(branch2)

    branch3 = AveragePooling2D((8, 8), (8, 8), 'same')(layer4)
    branch3 = convbn(branch3, 4, 1, 1, 1)
    branch3 = Activation('relu')(branch3)
    branch3 = UpSampling2DBilinear(layer4_size)(branch3)

    branch4 = AveragePooling2D((16, 16), (16, 16), 'same')(layer4)
    branch4 = convbn(branch4, 4, 1, 1, 1)
    branch4 = Activation('relu')(branch4)
    branch4 = UpSampling2DBilinear(layer4_size)(branch4)

    output_feature = concatenate(
        [layer2, layer4, branch4, branch3, branch2, branch1])
    lastconv = convbn(output_feature, 16, 3, 1, 1)
    lastconv = Activation('relu')(lastconv)
    lastconv = Conv2D(3, 1, (1, 1), 'same', use_bias=False)(lastconv)

    model = Model(inputs=[i], outputs=[lastconv])

    return model


def _getCostVolume_(inputs):
    shape = K.shape(inputs[0])
    disparity_values = np.linspace(-4, 4, 17)
    disparity_costs = []
    for d in disparity_values:
        if d == 0:
            tmp_list = []
            for i in range(len(inputs)):
                tmp_list.append(inputs[i])
        else:
            tmp_list = []
            for i in range(len(inputs)):
                (v, u) = divmod(i, 9)
                tensor = tfa.image.translate(inputs[i],
                                             [d * (u - 4), d * (v - 4)],
                                             'BILINEAR')
                tmp_list.append(tensor)

        cost = K.concatenate(tmp_list, axis=3)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1)
    cost_volume = K.reshape(cost_volume,
                            (shape[0], 17, shape[1], shape[2], 3 * 81))
    return cost_volume


def channel_attention(cost_volume):
    x = GlobalAveragePooling3D()(cost_volume)
    x = Lambda(
        lambda y: K.expand_dims(K.expand_dims(K.expand_dims(y, 1), 1), 1))(x)
    x = Conv3D(170, 1, 1, 'same')(x)
    x = Activation('relu')(x)
    x = Conv3D(15, 1, 1, 'same')(x)  # [B, 1, 1, 1, 15]
    x = Activation('sigmoid')(x)

    x = Lambda(lambda y: K.concatenate([
        y[:, :, :, :, 0:5], y[:, :, :, :, 1:2], y[:, :, :, :, 5:9],
        y[:, :, :, :, 2:3], y[:, :, :, :, 6:7], y[:, :, :, :, 9:12],
        y[:, :, :, :, 3:4], y[:, :, :, :, 7:8], y[:, :, :, :, 10:11],
        y[:, :, :, :, 12:14], y[:, :, :, :, 4:5], y[:, :, :, :, 8:9],
        y[:, :, :, :, 11:12], y[:, :, :, :, 13:15]
    ],
                                       axis=-1))(x)

    x = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 5, 5)))(x)
    x = Lambda(lambda y: tf.pad(y, [[0, 0], [0, 4], [0, 4]], 'REFLECT'))(x)
    attention = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 1, 1, 1, 81)))(x)
    x = Lambda(lambda y: K.repeat_elements(y, 3, -1))(attention)
    return multiply([x, cost_volume]), attention


def basic(cost_volume):

    feature = 2 * 75
    dres0 = convbn_3d(cost_volume, feature, 3, 1)
    dres0 = Activation('relu')(dres0)
    dres0 = convbn_3d(dres0, feature, 3, 1)
    cost0 = Activation('relu')(dres0)

    dres1 = convbn_3d(cost0, feature, 3, 1)
    dres1 = Activation('relu')(dres1)
    dres1 = convbn_3d(dres1, feature, 3, 1)
    cost0 = add([dres1, cost0])

    dres4 = convbn_3d(cost0, feature, 3, 1)
    dres4 = Activation('relu')(dres4)
    dres4 = convbn_3d(dres4, feature, 3, 1)
    cost0 = add([dres4, cost0])

    classify = convbn_3d(cost0, feature, 3, 1)
    classify = Activation('relu')(classify)
    cost = Conv3D(1, 4, 1, 'same', use_bias=False)(classify)

    return cost

def combine(cost, depth_cues):
    com = tf.concat([cost, depth_cues], axis=-1)
    return com 


def disparity_regression(inputs):
    shape = K.shape(inputs)
    disparity_values = np.linspace(-4, 4, 17)
    x = K.constant(disparity_values, shape=[17])
    x = K.expand_dims(K.expand_dims(K.expand_dims(x, 0), 0), 0)
    x = tf.tile(x, [shape[0], shape[1], shape[2], 1])
    out = K.sum(multiply([inputs, x]), -1)
    return out


def build_model(input_shape, angres):
    input_list = []

    for i in range(angres * angres):
        input_list.append(Input(shape=(input_shape[1], input_shape[2], 1)))
    feature_extraction_layer = feature_extraction(input_shape[1], input_shape[2])

    center = []
    feature_list = []
    n_range = range(35, 45)
    depth_cue_extractor = DepthCueExtractor()
    
    for i in range(angres * angres):
        f_map = feature_extraction_layer(input_list[i])
        if i in n_range:
            center = f_map
            depth_cues = depth_cue_extractor(center)
            feature_list.append(depth_cues)
        else:
            feature_list.append(f_map)

    ''' cost volume '''
    cv = Lambda(_getCostVolume_)(feature_list)
    ''' channel attention '''
    cv, attention = channel_attention(cv)
    cost = basic(cv)
    cost = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1),
                                                 (0, 2, 3, 1)))(cost)
    pred = Activation('softmax')(cost)
    pred = Lambda(disparity_regression)(pred)

    model = Model(inputs=input_list, outputs=[pred])
    model.summary()

    return model












