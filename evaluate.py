import numpy as np
import matplotlib.pyplot as plt
import model.model2 as net2
import load_data
import custom_metrics
import os
import plotly.express as px
import pandas as pd

data_path = '../../datasets'
path = data_path + '/hci_dataset/'
pred_path = 'predictions/'
pred_model = 'test14_val/'

def radar_plot(scores):
    '''
    method for creating a radar plot of median scores
    '''
    px.line_polar(scores, r='r', theta='theta', line_close=True) 

    
    

badpix7_dict = {} 
badpix3_dict = {}
badpix1_dict = {}
mse_dict = {}
hci_folder = [d.name for d in os.scandir(path) if d.is_dir()]
for sub_dir in hci_folder:
    scenes = [d.name for d in os.scandir(path + sub_dir) if d.is_dir()]
    for scene in scenes:
        r_dir = path + sub_dir + '/' + scene
        if 'stratified' not in r_dir and 'training' not in r_dir:
            continue
        gt = np.load(r_dir + '/stacked/center_disp.npy')
        gt = np.swapaxes(gt, 0, 1)
        pred = np.load(pred_path + pred_model + f'{scene}.npy')

        badpix7 = custom_metrics.badpix(y_pred=pred, y_true=gt, threshold=0.07)
        badpix7_map = custom_metrics.badpix_map(y_pred=pred, y_true=gt, threshold=0.07)
        badpix7_dict[scene] = badpix7
        np.save(pred_path + pred_model + f'{scene}_badpix.npy', badpix7_map)

        badpix3 = custom_metrics.badpix(y_pred=pred, y_true=gt, threshold=0.03)
        badpix3_dict[scene] = badpix3

        badpix1 = custom_metrics.badpix(y_pred=pred, y_true=gt, threshold=0.01)
        badpix1_dict[scene] = badpix1

        mse = custom_metrics.mse(pred, gt)
        mse_dict[scene] = mse

        diff_map = np.abs(pred-gt)
        np.save(pred_path + pred_model + f'{scene}_diff.npy', diff_map)


med_badpix7 = np.median(list(badpix7_dict.values()))
med_badpix3 = np.median(list(badpix3_dict.values()))
med_badpix1 = np.median(list(badpix1_dict.values()))
med_mse = np.median(list(mse_dict.values()))

# radar plot
df = pd.DataFrame(dict(
    r=[med_mse, med_badpix7, med_badpix3, med_badpix1],
    theta=['MSE', 'BadPix (0.07)', 'BadPix (0.03)', 'BadPix (0.01)']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
fig.update_traces(fill='toself')
fig.show()

print('Badpix threshold 0.07: ', badpix7_dict) 
print('mean: ', np.mean(list(badpix7_dict.values())))
print('median: ', med_badpix7)
print('Badpix threshold 0.03: ', badpix3_dict)
print('mean: ', np.mean(list(badpix3_dict.values())))
print('median: ', med_badpix3)
print('Badpix threshold 0.01: ', badpix1_dict)
print('mean: ', np.mean(list(badpix1_dict.values())))
print('median: ', med_badpix1)
print('MSE :', mse_dict)
print('mean: ', np.mean(list(mse_dict.values())))
print('median: ', med_mse)








