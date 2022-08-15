import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_dir = 'models/'

def plot_mse(model_name='model1'):
    load_dir = save_dir + model_name + '/'
    f = pd.read_csv(load_dir + 'history.csv')
    mse = f['val_mean_squared_error']
    plt.plot(mse)
    plt.ylabel('validation mean squared error')
    plt.xlabel('epoch')
    plt.show()

    mse = f['mean_squared_error']
    plt.plot(mse)
    plt.ylabel('training mean squared error')
    plt.xlabel('epoch')
    plt.show()
    
     
def plot_mae(model_name='model1'):
    load_dir = save_dir + model_name + '/'
    f = pd.read_csv(load_dir + 'history.csv')
    mse = f['val_mean_absolute_error']
    plt.plot(mse)
    plt.ylabel('validation mean absolute error')
    plt.xlabel('epoch')
    plt.show()

    mse = f['mean_absolute_error']
    plt.plot(mse)
    plt.ylabel('training mean absolute error')
    plt.xlabel('epoch')
    plt.show()



def plot_loss(model_name='model1'):
    load_dir = save_dir + model_name + '/'
    f = pd.read_csv(load_dir + 'history.csv')
    mse = f['val_loss']
    plt.plot(mse)
    plt.ylabel('validation loss')
    plt.xlabel('epoch')
    plt.show()

    mse = f['loss']
    plt.plot(mse)
    plt.ylabel('training mean loss')
    plt.xlabel('epoch')
    plt.show()






