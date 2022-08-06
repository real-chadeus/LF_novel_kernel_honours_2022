from PIL import Image, ImageChops
import numpy as np
import sys, glob, os, random
import pandas as pd
import matplotlib.pyplot as plt
import time

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


def proc_sai(img_path=None, img_array=None, img_size=420):
    '''
    returns subaperture image as numpy array, given the location of the image
    arg img_array: numpy array for when we need to transform disparity maps
    '''
    if img_path:
        img = Image.open(img_path)
    else:
        img = Image.fromarray(img_array)
    w, h = img.size
    # left, top, right, bottom
    w_offset = int((w-img_size)/2)
    h_offset = int((h-img_size)/2)
    l,r = w_offset, w_offset+img_size
    tp,b = h_offset, h_offset+img_size
    window = (l,tp,r,b)
    new_img = img.crop(window)
    #split img into red green blue components
    if img_path:
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
                    target_n_sai=49,
                    img_size = 420):
    '''
    flatten hci dataset SAIs into single LFI
    '''
    div = int(np.sqrt(target_n_sai)) #divisor to get the current subview
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
            to_shape=(div,img_size,div,img_size,3)
            lfi = np.zeros(to_shape, dtype=np.uint8)
            frames = []
            
        path = read_img_paths[i]
        frames.append(path)
        sai = proc_sai(img_path=path, img_size=img_size)
        # print(sai)
        u, v = (j-left)//div, (j-left)%div
        lfi[u,:,v,:,:] = sai

        j += 1
        if j == right:
            lfi = lfi.reshape((div*img_size, div*img_size, 3), order='F')
            new_img = Image.fromarray(lfi)
            #new_img.show()
            new_img.save(save_dir+name)
            print(f"{save_dir}{name} saved.")
            break

def flatten_sintel(save_dir,read_dir,
                    n_sai,name='stacked.png',
                    target_n_sai=49,
                    img_size=420):
    '''
    flatten sintel dataset
    this dataset contains 24 videos. Each video has 20-50 frames. Each frame has 81 (9x9) views.
    arg frame: frame number. Each frame corresponds to 1 full LFI
    arg target_n_sai: max 81, must be a perfect square
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_frames = len([f for f in os.scandir(read_dir + '04_04/') if f.is_file()])//2 #number of frames in the current scene
    print(n_frames, ' frames in this scene')

    for i in range(n_frames):
        if i < 10:
            frame = f"00{i}"
        else:
            frame = f"0{i}"

        view_x = 0 # x coordinate of the current subview
        view_y = 0 # y coordinate
        div = int(np.sqrt(target_n_sai)) #divisor to get the current subview
        left, right  = select_sai_range(n_sai)
        to_shape1=(div,img_size,div,img_size,3) #shape for images
        to_shape2=(div,img_size,div,img_size) #shape for disparity maps 
        lfi = np.zeros(to_shape1, dtype=np.uint8)
        disps = np.zeros(to_shape2, dtype=np.float32) # disparity maps combined
        print(f'left:{left}, right: {right}')

        for k in range(left, right+1):
            if k % 9 == 0 and k != 0:
                view_x += 1
                view_y = 0
            
            folder = f'0{view_x}_0{view_y}/'
            view_y += 1
            path = read_dir + folder + frame + '.' + img_format
            left, right  = select_sai_range(n_sai)
            sai = proc_sai(path, img_size=img_size)
            u, v = (k-left)//div, (k-left)%div
            lfi[u,:,v,:,:] = sai

            disp = np.load(read_dir + folder + frame + '.npy') # load disparity map for the given frame of the current view 
            disp = proc_sai(img_array=disp, img_size=img_size) # reshape
            disps[u,:,v,:] = disp # combine disparity maps in the same way as the images
            
            if folder == '04_04/':
                # reshapes disp map to (img_size, img_size) in the same way as the 2D image
                c_disp = np.load(read_dir + folder + frame + '.npy')
                c_disp = proc_sai(img_array=c_disp, img_size=img_size) 
                np.save(save_dir+f'{frame}_center.npy', c_disp)
                print(f"{save_dir}{frame}_center.npy saved.")

            if k == right:
                lfi = lfi.reshape((div*img_size, div*img_size, 3), order='F')
                new_img = Image.fromarray(lfi)
                #new_img.show()
                new_img.save(save_dir+frame +'_'+name)
                # stacked disparities
                np.save(save_dir+f'{frame}_stacked.npy', disps)
                print(f"{save_dir}{name} and {save_dir}{frame}_stacked.npy saved.")
        

if __name__ == "__main__":
    data_path = '../../../datasets'
    s = time.time()

    sintel_r_dirs = [d for d in os.scandir(data_path + '/Sintel_LF/Sintel_LFV_9x9_with_all_disp/') if d.is_dir()]
    for d in sintel_r_dirs:
        r_dir = d.path + '/'
        s_dir = r_dir + 'stacked/'
        print('read dir: ', r_dir)
        print('save dir: ', s_dir)
        flatten_sintel(save_dir = s_dir,
                        read_dir = r_dir,
                        n_sai=81, target_n_sai=16, img_size=512)

    hci_folder = [d for d in os.scandir(data_path + '/hci_dataset/') if d.is_dir()]
    for s in hci_folder:
        sub_dir = s.path
        hci_r_dirs = [d for d in os.scandir(sub_dir) if d.is_dir()]
        for d in hci_r_dirs:
            r_dir = d.path + '/'
            s_dir = r_dir + 'stacked/'
            print('read dir: ', r_dir)
            print('save dir: ', s_dir)
            flatten_hci(save_dir = s_dir, 
                            read_dir = r_dir,
                            n_sai = 80, target_n_sai=16, img_size = 512)
        
    e = time.time()
    print('time to flatten: ', e-s)
    











