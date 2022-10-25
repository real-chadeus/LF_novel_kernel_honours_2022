import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import model.model2 as net2
from PIL import Image
import load_data
import custom_metrics
from custom_metrics import BadPix
from preprocessing.hci_dataset_tools import file_io
import time

load_path = 'saved_models/'
input_shape = (9,512,512,9)
batch_size=1
model_name = 'test11_val'


class Timer(tf.keras.callbacks.Callback):
    def __init__(self):
        self.k = 0
        self.time = 0
        self.scene_list = ['backgammon', 'dots', 'pyramids', 'stripes', 
                                'boxes', 'cotton', 'dino', 'sideboard', 'bedroom', 
                                    'bicycle', 'herbs', 'origami']
        
    def on_batch_begin(self, batch, logs=None);
        self.time = time.time()
         
    def on_epoch_end(self, epoch, logs=None):
        curr_time = time.time()
        runtime = curr_time-time.time()
        print(f'runtime {self.scene_list[self.k]}: {runtime}')
        with open('predictions/' + model_name + f'/{self.scene_list[self.k]}.txt', 'w') as f:
            f.write(runtime)
        self.k += 1

gen = load_data.dataset_gen
hci_eval = tf.data.Dataset.from_generator(gen, 
                 args=(False, False, True, 32, False, 
                       True, 9, batch_size, 1000, False, True, True, True),
                        output_signature=(tf.TensorSpec(shape=(batch_size,) + input_shape, dtype=tf.float32)))
hci_test = tf.data.Dataset.from_generator(gen, 
                 args=(False, False, True, 32, False, 
                       True, 9, batch_size, 1000, False, False, True, True),
                        output_signature=(tf.TensorSpec(shape=(batch_size,) + input_shape, dtype=tf.float32)))

runtimeTimer = Timer()
model = keras.models.load_model(load_path + model_name, custom_objects={'BadPix': BadPix})
preds = model.predict(load_data.multi_input(hci_eval, test=True), workers=8, steps=8, callbacks=[runtimeTimer])
mse_list = []
badpix_list = []
gt_gen = load_data.disp_gen()
k = 0
scene_list = ['backgammon', 'dots', 'pyramids', 'stripes', 'boxes', 'cotton', 'dino', 'sideboard', 'bedroom', 'bicycle', 'herbs', 'origami']
for pred in preds:
    ground_truth = next(gt_gen)
    mse = custom_metrics.mse(pred, ground_truth) 
    badpix = custom_metrics.badpix(pred, ground_truth)
    mse_list.append(mse)
    badpix_list.append(badpix)
    np.save('predictions/' + model_name + f'/{scene_list[k]}.npy', pred)
    file_io.write_pfm(pred,  'predictions/' + model_name + f'/{scene_list[k]}.pfm')
    k += 1

preds = model.predict(load_data.multi_input(hci_test, test=True), workers=8, steps=4, callbacks=[runtimeTimer])
for pred in preds:
    np.save('predictions/' + model_name + f'/{scene_list[k]}.npy', pred)
    file_io.write_pfm(pred,  'predictions/' + model_name + f'/{scene_list[k]}.pfm')
    k += 1


print('MSE LIST ', mse_list)
print('BADPIX LIST ', badpix_list)
print('MEAN MSE ', np.mean(mse_list))
print('MEAN BADPIX ', np.mean(badpix)) 













