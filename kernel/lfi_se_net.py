from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf

def allocate_gpu_memory():
    # for single GPU setups
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if physical_devices:
        try:
            print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.experiemntal.get_memory_info(physical_devices[0])
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU {} memory is allocated".format(0))
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected")


#def build_model(): 
    
     




