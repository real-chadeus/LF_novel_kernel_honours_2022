import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import model.model as se_net
from PIL import Image
import load_data
from custom_metrics import BadPix

load_path = 'saved_models/'

input_shape = (9,32,32,9)
batch_size=1

gen = load_data.dataset_gen
hci = tf.data.Dataset.from_generator(gen, 
                 args=(False, False, True, 32, False, 
                       True, 9, batch_size, 1000, False, False, True),
                        output_signature=(tf.TensorSpec(shape=(batch_size,) + input_shape, dtype=tf.float32)))

custom_metrics = {'BadPix7': BadPix(threshold=0.07), 'BadPix3': BadPix(threshold=0.03), 'BadPix1': BadPix(threshold=0.01)}
model = keras.models.load_model(load_path + 'test5', custom_objects={'BadPix': BadPix})
predictions = model.predict(hci, workers=8)

for i in range(predictions.shape[0]):
    for k in range(4):
        stacked = [[] for i in range(16)]
        for y in range(16):
            for x in range(16):
                stacked[y].append(predictions[(256 * k) + (y*16 + x)])
                

        stacked = np.block(stacked)
        np.save(f'predictions/pred_{k}.npy', stacked)














