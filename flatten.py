from PIL import Image, ImageChops
import numpy as np
import sys, glob, os, random
import pandas as pd

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



def proc_sai(p):
    img = Image.open(p)
    w, h = img.size
    # left, top, right, bottom
    w_offset = int((w-434)/2)
    h_offset = int((h-434)/2)
    l,r = w_offset, w_offset+434
    tp,b = h_offset, h_offset+434
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



def flatten_dataset(save_dir,read_dir,
                    n_sai,name='stacked.png',
                    target_n_sai=49):
    read_dir = read_dir
    read_img_paths = glob.glob(read_dir+"**/*."+img_format, recursive=True)
    read_img_paths = sorted(read_img_paths)

    save_dir = './'
    save_ref = save_dir+'/ref/'
    save_dist = save_dir+'/dist/'
    if not os.path.exists(save_ref):
        os.makedirs(save_ref)

    if not os.path.exists(save_dist):
        os.makedirs(save_dist)
        ref_word = 'ALL_REF'


    # print(read_img_paths)
    print(len(read_img_paths)//101)
    left, right  = select_sai_range(n_sai)

    print(f'left:{left}, right: {right}')

    i = 0
    j = 0

    for i in range(len(read_img_paths)):
        if not (left <= j <= right):
            j += 1
            continue

        if j == left:
            to_shape=(7,434,7,434,3)
            lfi = np.zeros(to_shape, dtype=np.uint8)
            frames = []
            
        p = read_img_paths[i]
        frames.append(p)

        sai = proc_sai(p)
        # print(sai)
        u, v = (j-left)//7, (j-left)%7
        lfi[u,:,v,:,:] = sai

        j += 1
        if j == right:
            # print(frames)
            j = 0
            parts = p.split(s)
            lfi = lfi.reshape((7*434, 7*434, 3), order='F')

            new_img = Image.fromarray(lfi)

            new_img.save(save_ref+name,img_format)
            
            print(f"{name} saved.")
            break


if __name__ == "__main__":
    data_path = '../../datasets'
    flatten_dataset(save_dir=data_path + '/hci_dataset/training/boxes/stacked', 
                    read_dir=data_path + '/hci_dataset/training/boxes/',
                    n_sai=80)