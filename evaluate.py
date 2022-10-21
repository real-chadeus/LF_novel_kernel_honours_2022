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

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

load_path = 'saved_models/'
input_shape = (9,512,512,9)
batch_size=1

gen = load_data.dataset_gen
hci = tf.data.Dataset.from_generator(gen, 
                 args=(False, False, True, 32, False, 
                       True, 9, batch_size, 1000, False, True, True, True),
                        output_signature=(tf.TensorSpec(shape=(batch_size,) + input_shape, dtype=tf.float32),
                                          tf.TensorSpec(shape=(batch_size,) + (input_shape[1], input_shape[2]), dtype=tf.float32)))

model = keras.models.load_model(load_path + 'test6_val', custom_objects={'BadPix': BadPix})
preds = model.predict(load_data.multi_input(hci, test=True), workers=8, steps=8)

mse_list = []
badpix_list = []

gt_gen = load_data.disp_gen()
for pred in preds:
    ground_truth = next(gt_gen)
    mse = custom_metrics.mse(pred, ground_truth) 
    badpix = custom_metrics.badpix(pred, ground_truth)
    mse_list.append(mse)
    badpix_list.append(badpix)

print('MSE LIST ', mse_list)
print('BADPIX LIST ', badpix_list)
print('MEAN MSE ', np.mean(mse_list))
print('MEAN BADPIX ', np.mean(badpix)) 











