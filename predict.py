import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import kernel.lfi_se_net as se_net
import preprocessing.flatten as flatten
import preprocessing.hci_dataset_tools.file_io as hci_io
import load
from PIL import Image

load_path = 'saved_models/'

input_shape = (3,436,3,436,3)
hci = functools.partial(load_data.dataset_gen, 
                            load_sintel=False, load_hci=True, crop=False, window_size=32,
                            augment_sintel=False, augment_hci=False,
                            batch_size=1)
hci = tf.data.Dataset.from_generator(hci,
      output_signature=(tf.TensorSpec(shape=(1,) + input_shape, dtype=tf.int8),
                        tf.TensorSpec(shape=(1,) + (input_shape[1], input_shape[2]), dtype=tf.float32)))
model = keras.models.load_model(load_path + 'hci_only5')
predictions = model.predict(hci, batch_size=1, workers=8)

for i in range(predictions.shape[0]):
    print(predcitions[i].shape)
    disp_map = 
    np.save(f'predictions/pred_{i}.npy', predictions[i])














