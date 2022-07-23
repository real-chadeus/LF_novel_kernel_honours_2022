from PIL import Image, ImageChops
import numpy as np
import sys, glob, os, random
import pandas as pd
import matplotlib.pyplot as plt

### Flatten datasets which have LFIs as 2D image files of each SAI.
exclude_ref = True
dataset_root, _, s = ".\\", None, '\\'
dn = "MPI-LFA"
af = "x8" ###-
img_format = "png"

def select_sai_range(n_sai, target_n_sai=49):
    n_sai = n_sai
    target_n_sai = target_n_sai
    mid = n_sai//2
    left = mid - target_n_sai//2
    right = mid + target_n_sai//2
    return left, right


def proc_sai(img_path, img_size=512):
    '''
    returns subaperture image as numpy array, given the location of the image
    '''
    img = Image.open(img_path)
    w, h = img.size
    # left, top, right, bottom
    w_offset = int((w-img_size)/2)
    h_offset = int((h-img_size)/2)
    l,r = w_offset, w_offset+img_size
    tp,b = h_offset, h_offset+img_size
    window = (l,tp,r,b)
    new_img = img.crop(window)
    #split img into red green blue components
    r, g, b = img.split()
    r = r.crop(window)
    g = g.crop(window)
    b = b.crop(window)
    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)
    new_img = np.asarray([r,g,b])
    new_img = np.moveaxis(new_img, 0, -1)
    return np.asarray(new_img)


def flatten_hci(save_dir,read_dir,
                    n_sai,name='stacked.png',
                    target_n_sai=49):
    '''
    flatten hci dataset SAIs into single LFI
    '''
    read_img_paths = glob.glob(read_dir+"**/*."+img_format, recursive=True)
    read_img_paths = sorted(read_img_paths)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    left, right  = select_sai_range(n_sai)
    print(f'left:{left}, right: {right}')
    i = 0
    j = 0

    for i in range(len(read_img_paths)):
        if not (left <= j <= right):
            j += 1
            continue

        if j == left:
            to_shape=(7,512,7,512,3)
            lfi = np.zeros(to_shape, dtype=np.uint8)
            frames = []
            
        path = read_img_paths[i]
        frames.append(path)
        sai = proc_sai(path)
        # print(sai)
        u, v = (j-left)//7, (j-left)%7
        lfi[u,:,v,:,:] = sai

        j += 1
        if j == right:
            lfi = lfi.reshape((7*512, 7*512, 3), order='F')
            new_img = Image.fromarray(lfi)
            new_img.show()
            print(save_dir)
            new_img.save(save_dir+name)
            print(f"{name} saved.")
            break

def flatten_sintel(save_dir,read_dir,
                    n_sai,name='stacked.png',
                    target_n_sai=49, frame='000'):
    '''
    flatten sintel dataset
    this dataset contains 24 videos. Each video has 20-50 frames. Each frame has 81 (9x9) views.
    arg frame: frame number. Each frame corresponds to 1 full LFI
    arg target_n_sai: max 81, must be a perfect square
    '''


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    div = int(np.sqrt(target_n_sai)) #divisor to get the current subview
    view_x = 0 # x coordinate of the current subview
    view_y = 0 # y coordinate
    num_files = len([name for name in os.listdir(read_dir + '04_04/') if os.path.isfile(name)])/2
    left, right  = select_sai_range(n_sai)
    to_shape=(7,512,7,512,3)
    lfi = np.zeros(to_shape, dtype=np.uint8)

    for k in range(left, right+1):
        if k % 9 == 0 and k != 0:
            view_x += 1
            view_y = 0
        
        folder = f'0{view_x}_0{view_y}/'
        path = read_dir + folder + frame + '.' + img_format
        left, right  = select_sai_range(n_sai)
        print(f'left:{left}, right: {right}')
        sai = proc_sai(path)
        u, v = (k-left)//div, (k-left)%div
        lfi[u,:,v,:,:] = sai

        if k == right:
            lfi = lfi.reshape((7*512, 7*512, 3), order='F')
            new_img = Image.fromarray(lfi)
            new_img.show()
            print(save_dir)
            new_img.save(save_dir+name)
            break
        
        

if __name__ == "__main__":
    data_path = '../../../datasets'
    #flatten_hci(save_dir=data_path + '/hci_dataset/training/boxes/stacked/', 
    #                read_dir=data_path + '/hci_dataset/training/boxes/',
    #                n_sai=80)
    
    flatten_sintel(save_dir = data_path + '/Sintel_LF/Sintel_LFV_9x9_with_all_disp/ambushfight_1/stacked/',
                    read_dir = data_path + '/Sintel_LF/Sintel_LFV_9x9_with_all_disp/ambushfight_1/',
                    n_sai=81)







