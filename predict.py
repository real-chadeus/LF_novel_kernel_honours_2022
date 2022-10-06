import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import model.model as se_net
from PIL import Image

load_path = 'saved_models/'

input_shape = (3,436,3,436,3)

    gen = load_data.dataset_gen
    val = tf.data.Dataset.from_generator(gen, 
                     args=(False, False, True, 32, False, 
                           True, 9, batch_size, 1000, False, False, True),
                            output_signature=(tf.TensorSpec(shape=(batch_size,) + input_shape, dtype=tf.float32),
                                              tf.TensorSpec(shape=(batch_size,) + (input_shape[1], input_shape[2]), dtype=tf.float32)))
model = keras.models.load_model(load_path + 'test5')
predictions = model.predict(hci, batch_size=1, workers=8)

for i in range(predictions.shape[0]):
    print(predcitions[i].shape)
    np.save(f'predictions/pred_{i}.npy', predictions[i])














