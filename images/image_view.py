from PIL import Image, ImageChops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = '../../../datasets'

d_map = np.load(data_path + '/hci_dataset/additional/antinous/stacked/center_disp.npy')
#plt.imshow(d_map, interpolation='nearest')
#plt.show()
#plt.imshow(np.swapaxes(d_map, 0,1), interpolation='nearest')
#plt.show()
img = Image.open(data_path + '/hci_dataset/additional/vinyl/stacked/stacked.png')
img = np.asarray(img)
img = img.reshape((9,512,9,512,3), order='F')
img = np.moveaxis(img, 2, 3)
img = img/255
img = 0.2126 * img[:, :, :, :, 0] + 0.7152 * img[:,:,:,:,1] + 0.0722 * img[:,:,:,:,2]
center1 = img[4,:,:,4]
plt.imshow(center1)
plt.show()
d_map = np.load(data_path + '/hci_dataset/additional/vinyl/stacked/center_disp.npy')
d_map = np.swapaxes(d_map, 0, 1)
plt.imshow(d_map, interpolation='nearest')
plt.show()
#for x in range(15):
#    for y in range(15): 
#        crop_img = img[4, 32*x:32*(x+1), 32*y:32*(y+1), 4, :]
#        crop_map = d_map[32*x:32*(x+1),32*y:32*(y+1)]
#        print(crop_map)
#        plt.imshow(crop_img)
#        plt.show()


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









