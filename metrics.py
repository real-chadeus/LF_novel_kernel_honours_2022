import numpy as np
import pathlib, datetime
import load
import preprocessing.hci_dataset_tools.file_io as hci_io
import os
import argparse

def BadPix(gt, algo, threshold=0.07):
    '''
    calculates the percentage of pixels at the given mask with
    abs(gt-algo) > threshold
    ''' 
    gt = np.asarray(gt)
    algo = np.asarray(algo)   
    diff_map = np.absolute(np.subtract(gt, algo))
    n_badpix = np.sum(diff_map > threshold)
    badpix = n_badpix/diff_map.size 
    
    return badpix
     
    




