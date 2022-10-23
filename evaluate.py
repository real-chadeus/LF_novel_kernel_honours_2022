import numpy as np
import matplotlib.pyplot as plt
import model.model2 as net2
import load_data
import custom_metrics
import os

data_path = '../../datasets'
path = data_path + '/hci_dataset/'
pred_path = 'predictions/'
pred_model = 'test11_val/'

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
        badpix7_dict[scene] = badpix7

        badpix3 = custom_metrics.badpix(y_pred=pred, y_true=gt, threshold=0.03)
        badpix3_dict[scene] = badpix3

        badpix1 = custom_metrics.badpix(y_pred=pred, y_true=gt, threshold=0.01)
        badpix1_dict[scene] = badpix1

        mse = custom_metrics.mse(pred, gt)
        mse_dict[scene] = mse

        diff_map = np.abs(pred-gt)
        np.save(pred_path + pred_model + f'{scene}_diff.npy', diff_map)

print('Badpix threshold 0.07: ', badpix7_dict) 
print('mean: ', np.mean(list(badpix7_dict.values())))
print('Badpix threshold 0.03: ', badpix3_dict)
print('mean: ', np.mean(list(badpix3_dict.values())))
print('Badpix threshold 0.01: ', badpix1_dict)
print('mean: ', np.mean(list(badpix1_dict.values())))
print('MSE :', mse_dict)
print('mean: ', np.mean(list(mse_dict.values())))








