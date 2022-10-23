import numpy as np
import tensorflow as tf
import pathlib, datetime
import load_data
import preprocessing.hci_dataset_tools.file_io as hci_io
import os
import argparse

class BadPix(tf.keras.metrics.Metric):
    def __init__(self, name='BadPix7', threshold=0.07, **kwargs):
        super(BadPix, self).__init__(name=name)
        self.badpix = 0 
        self.threshold = tf.constant(threshold, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        calculates the percentage of pixels at the given mask with
        abs(y_true-y_pred) > threshold
        ''' 
        diff_map = tf.math.abs(tf.math.subtract(y_true, y_pred))
        n_badpix = tf.math.reduce_sum(tf.where(diff_map > self.threshold, 1, 0))
        badpix = n_badpix/tf.size(y_true) 
        self.badpix = badpix

    def get_config(self):
        config = super().get_config()
        return config
    
    @classmethod 
    def from_config(cls, config):
        return cls(**config)
        
    def result(self):
        return self.badpix 


def badpix(y_pred, y_true, threshold=0.07):
    diff = np.abs(y_pred-y_true)
    bp = np.sum(np.where(diff > threshold, 1, 0)) 
    bp = bp/np.size(y_true)
    return bp

def mse(y_pred, y_true):
    mse = ((y_pred - y_true) ** 2).mean()
    return mse




