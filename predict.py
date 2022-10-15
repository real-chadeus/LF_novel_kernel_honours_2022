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
model_name = 'test7_val'

gen = load_data.dataset_gen
hci = tf.data.Dataset.from_generator(gen, 
                 args=(False, False, True, 32, False, 
                       True, 9, batch_size, 1000, False, True, True, True),
                        output_signature=(tf.TensorSpec(shape=(batch_size,) + input_shape, dtype=tf.float32),
                                           tf.TensorSpec(shape=(batch_size,) + (input_shape[1], input_shape[2]))))

model = keras.models.load_model(load_path + 'test7_val', custom_objects={'BadPix': BadPix})
predictions = model.predict(load_data.multi_input(hci, test=True), workers=8, steps=12)

k = 0
for pred in predictions:
    np.save('predictions/' + model_name + f'/pred_{k}.npy', pred)
    k += 1








