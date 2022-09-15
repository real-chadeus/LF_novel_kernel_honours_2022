from PIL import Image, ImageChops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = '../../../datasets'

d_map = np.load(data_path + '/hci_dataset/additional/pens/stacked/center_disp.npy')
plt.imshow(d_map, interpolation='nearest')
plt.show()
plt.imshow(np.swapaxes(d_map, 0,1), interpolation='nearest')
plt.show()
img = Image.open(data_path + '/hci_dataset/additional/pens/stacked/stacked.png')
img = np.asarray(img)
img = img.reshape((9,512,9,512,3), order='F')
center1 = img[4, :, 4, :, :]
center_red = img[4, :, 4, :, 0]
center_green = img[4, :, 4, :, 1]
center_blue = img[4, :, 4, :, 2]
plt.imshow(center1)
plt.show()
plt.imshow(center_red)
plt.show()
plt.imshow(center_green)
plt.show()
plt.imshow(center_blue)
plt.show()


#d_map = np.load(data_path + '/Sintel_LF/Sintel_LFV_9x9_with_all_disp/chickenrun_3/stacked/000_center.npy')
#plt.imshow(d_map, interpolation='nearest')
#plt.show()
#plt.imshow(np.swapaxes(d_map, 0,1), interpolation='nearest')
#plt.show()


#img = Image.open(data_path + '/hci_dataset/additional/tomb/stacked/stacked.png')
#img = np.asarray(img)
#img = img.reshape((9,512,9,512,3), order='F')
#lfi = []
#for i in range(9):
#    for k in range(9):
#        lfi.append(img[i, :, k, :, :])
#lfi = np.asarray(lfi)
#lfi = np.stack(lfi)
##img = np.moveaxis(img, 2,3)
##img = np.moveaxis(img, 0,2) 
##print(img.shape)
#print(lfi.shape)
#center = lfi[46, :, :, :] 
#plt.imshow(center)
#plt.show()


#img = Image.open(data_path + '/Sintel_LF/Sintel_LFV_9x9_with_all_disp/ambushfight_1/stacked/000_stacked.png')
#img = np.asarray(img)
#img = img.reshape((9,512,9,512,3), order='F')
#lfi = []
#for i in range(9):
#    for k in range(9):
#        lfi.append(img[i, :, k, :, :])
#lfi = np.asarray(lfi)
#lfi = np.stack(lfi)
##img = np.moveaxis(img, 2,3)
##img = np.moveaxis(img, 0,2) 
##print(img.shape)
#print(lfi.shape)
#center = lfi[46, :, :, :] 
#plt.imshow(center)
#plt.show()









