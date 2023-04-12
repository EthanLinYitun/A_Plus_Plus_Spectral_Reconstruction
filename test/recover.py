# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:11:51 2021
"""

import utils
import os
from glob import glob
import numpy as np
import time


test_modes = ['orig', 'rot', 'blur10', 'blur20']

# recover hyperspectral images from RGB images and store them
# comment out other SR methods if only want to run one of them
for mode in test_modes:
    # load file paths of RGB images
    rgb_files = glob('./data/rgb/'+mode+'/*.npy')
    print('#### Now recover the', mode, 'images ####')
    
    # split_axis setting for the DNNs
    if mode in ['rot']:
        split_axis = 2
    else:
        split_axis = 3
    
    # iterate through the RGB files
    for file_path in rgb_files:
        _, file_name_ext = os.path.split(file_path)
        file_name, _ = os.path.splitext(file_name_ext)
        print('Current image:', file_name)
        
        # load rgb file
        rgb = np.load(file_path)
        
        ## A++ recovery
        print('      A++ method')
        model = utils.load_a_plus_plus_model()
        t = time.time()
        recovery = utils.recover_a_plus_plus(rgb, model[0], model[1])
        print('      (', time.time()-t, 'seconds )')
        np.save(os.path.join('./data/hyperspectral_rec/A++/'+mode, file_name), recovery)
        
        
        ## A+ recovery
        print('      A+ method')
        model = utils.load_a_plus_model()
        t = time.time()
        recovery = utils.recover_a_plus(rgb, model[0], model[1])
        print('      (', time.time()-t, 'seconds )')
        np.save(os.path.join('./data/hyperspectral_rec/A+/'+mode, file_name), recovery)
    
        
        ## PR-RELS recovery
        print('      PR-RELS method')
        model = utils.load_pr_rels()
        t = time.time()
        recovery = utils.recover_pr_rels(rgb, model)
        print('      (', time.time()-t, 'seconds )')
        np.save(os.path.join('./data/hyperspectral_rec/PR-RELS/'+mode, file_name), recovery)    
        
        
        ## AWAN recovery
        print('      AWAN method')
        model = utils.load_AWAN_model('orig')
        t = time.time()
        recovery = utils.recover_AWAN(rgb, model, split_axis)
        print('      (', time.time()-t, 'seconds )')
        np.save(os.path.join('./data/hyperspectral_rec/AWAN/'+mode, file_name), recovery)    
        
        
        ## AWAN-aug3 recovery
        print('      AWAN-aug3 method')
        model = utils.load_AWAN_model('aug')
        t = time.time()
        recovery = utils.recover_AWAN(rgb, model, split_axis)
        print('      (', time.time()-t, 'seconds )')
        np.save(os.path.join('./data/hyperspectral_rec/AWAN-aug3/'+mode, file_name), recovery)    
        
        
        ## HSCNN-D recovery
        print('      HSCNN-D method')
        model = utils.load_HSCNN_D_model()
        t = time.time()
        recovery = utils.recover_HSCNN_D(rgb, model, split_axis)
        print('      (', time.time()-t, 'seconds )')
        np.save(os.path.join('./data/hyperspectral_rec/HSCNN-D/'+mode, file_name), recovery)    