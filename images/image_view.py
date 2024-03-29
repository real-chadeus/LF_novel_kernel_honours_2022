from PIL import Image, ImageChops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


data_path = '../../../datasets'
val_model = 'test14_val'

#img = Image.open(data_path + '/hci_dataset/training/sideboard/stacked/stacked.png')
#img = np.asarray(img)
#img = img.reshape((9,512,9,512,3), order='F')
#img = np.moveaxis(img, 2, 3)
#img = img/255
#img = 0.2126 * img[:, :, :, :, 0] + 0.7152 * img[:,:,:,:,1] + 0.0722 * img[:,:,:,:,2]
#img = np.flip(img, axis=2)
#center1 = img[4,:,:,4]
#plt.imshow(center1)
#plt.show()
d_map = np.load(data_path + '/hci_dataset/stratified/backgammon/stacked/center_disp.npy')
d_map = np.swapaxes(d_map, 0, 1)
d_map = np.flip(d_map, axis=1)
plt.imshow(d_map)
plt.show()

preds = np.load('../predictions/' + val_model + '/bedroom.npy')
#plt.imshow(preds, cmap='inferno')
plt.imshow(preds)
plt.show()

#stacked = [[] for i in range(16)]
#for y in range(16):
#    for x in range(16): 
#        crop_map = d_map[32*x:32*(x+1),32*y:32*(y+1)]
#        stacked[x].append(crop_map)
#stacked = np.block(stacked)
#plt.imshow(stacked, interpolation='nearest')
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
















