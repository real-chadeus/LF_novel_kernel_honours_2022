import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import model.model2 as net2
from PIL import Image
import load_data
from custom_metrics import BadPix

load_path = 'saved_models/'
input_shape = (9,512,512,9)
batch_size=1

gen = load_data.dataset_gen
hci = tf.data.Dataset.from_generator(gen, 
                 args=(False, False, True, 32, False, 
                       True, 9, batch_size, 1000, False, True, True, True),
                        output_signature=(tf.TensorSpec(shape=(batch_size,) + input_shape, dtype=tf.float32)))

custom_metrics = {'BadPix7': BadPix(threshold=0.07), 'BadPix3': BadPix(threshold=0.03), 'BadPix1': BadPix(threshold=0.01)}
model = keras.models.load_model(load_path + 'test6_val', custom_objects={'BadPix': BadPix})
predictions = model.predict(load_data.multi_input(hci, test=True), workers=8, steps=12)

k = 0
for pred in predictions:
    np.save(f'predictions/pred_{k}.npy', pred)
    k += 1



