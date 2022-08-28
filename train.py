import pathlib, datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
import tensorflow_addons as tfa
import load_data
import preprocessing.hci_dataset_tools.file_io as hci_io
import kernel.lfi_se_net as se_net
import tensorflow.keras.losses as losses
import numpy as np
from tqdm.keras import TqdmCallback
import os
import argparse
from custom_metrics import BadPix
import time
import functools

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
print('tensorflow version: ', tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
sintel_folders = ['../../datasets/Sintel_LF/Sintel_LFV_9x9_with_all_disp/ambushfight_1']
#tf.config.set_logical_device_configuration(
#    physical_devices[0],
#    [tf.config.LogicalDeviceConfiguration(memory_limit=8500)])
save_path = 'models/'

def step_decay(epoch):
    # learning rate schedule
    factor = 1
    if epoch >= 10: factor = 0.1
    if epoch >= 15: factor = 0.01
    return lr * factor


def train(model, input_shape=(), dataset=(), val_set=[], 
            epochs=10, batch_size=1, model_name='model1', 
            use_gen=True, load_model=False):
    '''
    train function
    arg dataset: 2-tuple of data, first element = train data, second element = validation data.
                 Each is a 2-tuple of (data, labels) 
    arg use_gen: use generators to reduce memory usage
    '''
    if not os.path.exists(save_path + model_name):
        os.makedirs(save_path + model_name)

    lr = 0.0001
    loss = losses.MeanAbsoluteError()
    optimizer = Adam(learning_rate=lr)
    # model compile
    model.compile(optimizer=optimizer, loss=loss, 
                   metrics=[tf.keras.metrics.MeanSquaredError(),
                            BadPix(name='BadPix7', threshold=0.07),
                            BadPix(name='BadPix3', threshold=0.03),
                            BadPix(name='BadPix1', threshold=0.01)
                            ])

    # checkpoint
    checkpoint = ModelCheckpoint(filepath = save_path + model_name, monitor='val_mean_squared_error',
            save_best_only=True, save_weights_only=False, verbose=1, mode='auto')
    # callbacks
    logger = CSVLogger(save_path + model_name + '/history.csv', separator=',')
    tqdm_callback = tfa.callbacks.TQDMProgressBar()
    #lr_schedule = LearningRateScheduler(step_decay, verbose=1)

    if load_model:
        model = keras.models.load_model(save_path + model_name, custom_objects={'BadPix': BadPix})

    # train model
    if use_gen:
        val = val_set 
        gen = functools.partial(load_data.dataset_gen, input_shape)
        #def data_gen(): 
        #    for i in range(train_data.shape[0]):
        #        yield train_data[i], train_labels[i]  
        training = tf.data.Dataset.from_generator(gen,
              output_signature=(tf.TensorSpec(shape=(1,) + input_shape, dtype=tf.int8),
                                tf.TensorSpec(shape=(input_shape[1], input_shape[3]), dtype=tf.float32)))

        model.fit(x=training,batch_size=batch_size, 
                    epochs=epochs, validation_data=val,
                    validation_batch_size=1, 
                    callbacks=[TqdmCallback(verbose=2), 
                                checkpoint, logger])
    else:
        train = dataset[0]
        train_data = train[0]
        train_labels = train[1]
        val = dataset[1]
        model.fit(x=train_data, y=train_labels, batch_size=batch_size, 
                    epochs=epochs, validation_data=val,
                    validation_batch_size=1,
                    callbacks=[TqdmCallback(verbose=2), logger])

    model.save(save_path + model_name)


if __name__ == "__main__":
    # define model
    #input_shape = (7,512,7,512,3)
    input_shape = (9,436,9,436,3)
    model = se_net.build_model(input_shape=input_shape, summary=True, 
                                n_sais=81)
    # load datasets
    hci_val = load_data.load_hci(img_shape=input_shape, do_augment=False, 
                                use_tf_ds=False, use_disp=True)
    
    # training
    start = time.time()
    train(model=model, input_shape=input_shape, batch_size=32, 
            val_set=hci_val, epochs=10, model_name='model5', 
            use_gen=True, load_model=False)
    end = time.time()
    print('time to train: ', end-start)












