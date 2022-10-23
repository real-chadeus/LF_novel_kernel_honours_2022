import pathlib, datetime
import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import tensorflow_addons as tfa
import load_data
import preprocessing.hci_dataset_tools.file_io as hci_io
import model.model2 as net2
import tensorflow.keras.losses as losses
from keras import backend as K
import numpy as np
from tqdm.keras import TqdmCallback
import os
import argparse
import custom_metrics
from custom_metrics import BadPix
import time
import functools

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
print('tensorflow version: ', tf.__version__)
tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
save_path = 'saved_models/'

class MemoryCleaner(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def train(input_shape=(), val_shape=(), dataset=(), 
            epochs=10, batch_size=1, model_name='model1', save_model='model1', 
            use_gen=True, load_model=False, load_sintel=True,
            load_hci=True, augment_sintel=True, augment_hci=True,
            crop=True, window_size=32):
    '''
    train function
    '''
    if not os.path.exists(save_path + model_name):
        os.makedirs(save_path + model_name)

    if load_model:
        print('loading...')
        model = keras.models.load_model(save_path + model_name, custom_objects={'BadPix': BadPix})
        val_model = keras.models.load_model(save_path + model_name + '_val', custom_objects={'BadPix': BadPix})
    else:
        model = net2.build_model(input_shape=input_shape, angres=9)
        val_model = net2.build_model(input_shape=val_shape, angres=9)

    lr = 0.001
    #loss = losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # model compile
    model.compile(optimizer=optimizer, loss='mae', 
                   metrics=[tf.keras.metrics.MeanSquaredError(),
                            BadPix(name='BadPix7', threshold=0.07),
                            BadPix(name='BadPix3', threshold=0.03),
                            BadPix(name='BadPix1', threshold=0.01)
                            ])

    val_model.compile(optimizer=optimizer, loss='mae', 
                   metrics=[tf.keras.metrics.MeanSquaredError(),
                            BadPix(name='BadPix7', threshold=0.07),
                            BadPix(name='BadPix3', threshold=0.03),
                            BadPix(name='BadPix1', threshold=0.01)
                            ])

    # callbacks
    logger = CSVLogger(save_path + model_name + '/history.csv', separator=',')
    memory_cleaner = MemoryCleaner()

    # validation dataset
    gen = load_data.dataset_gen 
    val_set = tf.data.Dataset.from_generator(gen, 
                     args=(False, False, True, 32, False, 
                           True, 9, batch_size, 1000, False, True, True, True),
                            output_signature=(tf.TensorSpec(shape=(batch_size,) + val_shape, dtype=tf.float32)))

    # training dataset
    train_set = tf.data.Dataset.from_generator(gen, 
                     args=(augment_sintel, augment_hci, True, 32, load_sintel, 
                           load_hci, 9, batch_size, 1000, True, False, False, False),
                            output_signature=(tf.TensorSpec(shape=(batch_size,) + input_shape, dtype=tf.float32),
                                              tf.TensorSpec(shape=(batch_size,) + (input_shape[1], input_shape[2]), dtype=tf.float32)))

    #training
    best_badpix=0.057
    for i in range(epochs):
        print(f'epoch {i+1} of {epochs} starting')
        gc.collect()
        tf.keras.backend.clear_session()
        model.fit(x=load_data.multi_input(train_set), epochs=1, steps_per_epoch=5000, 
                    callbacks=[TqdmCallback(verbose=2), 
                                logger, memory_cleaner],
                                workers=8)
        
        weights = model.get_weights()
        val_model.set_weights(weights)

        preds = val_model.predict(load_data.multi_input(val_set, test=True), workers=8, steps=8)
        mse_list = []
        badpix_list = []

        gt_gen = load_data.disp_gen()
        for pred in preds:
            ground_truth = next(gt_gen)
            mse = custom_metrics.mse(pred, ground_truth) 
            badpix = custom_metrics.badpix(pred, ground_truth)
            mse_list.append(mse)
            badpix_list.append(badpix)
        mean_badpix = np.mean(badpix_list)
        mean_mse = np.mean(mse_list)

        print(f'full model evaluation.   mean MSE: {mean_mse}, mean badpix: {mean_badpix}\nfull mse: {mse_list}\n full badpix: {badpix_list}')
        print('previous best badpix ', best_badpix)
        update = 0
        if mean_badpix < best_badpix:
            update = 1
            model.save(save_path + save_model)
            val_model.save(save_path + save_model + '_val')
            best_badpix = mean_badpix
            print('current best badpix ', best_badpix)

        gc.collect()
        tf.keras.backend.clear_session()

        if i == epochs-1 and update == 0:
            model.save(save_path + save_model)
            val_model.save(save_path + save_model + '_val')

        update = 0
        print(f'epoch {i+1} of {epochs} finished')


if __name__ == "__main__":
   
    # initial parameters 
    batch_size = 1
    input_shape = (9, 128, 128, 9)
    val_shape = (9, 512, 512, 9)

    # training
    start = time.time()
    train(input_shape=input_shape, val_shape=val_shape, batch_size=batch_size,  
            epochs=3, model_name='test11', save_model='test11', use_gen=True, load_model=True, 
            load_sintel=False, load_hci=True, augment_sintel=True, augment_hci=True)
    end = time.time()
    print('time to train: ', end-start)






