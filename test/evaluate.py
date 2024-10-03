# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:40:22 2021
"""

from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from rgb_simulate import load_hyperspectral_data


def mrae(gt, rec):
    gt = gt.reshape(31, -1).T
    rec = rec.reshape(31, -1).T
    
    return np.mean(np.abs(gt - rec) / gt, axis=1)


def show_error_map(mrae_eval, model_name):
    fig = plt.figure()
    plt.axis('off')
    fig.suptitle(file_name+' '+model_name+' recovery error', fontsize=16)
    cost_img = mrae_eval.reshape(gt.shape[1], gt.shape[2])
    plt.imshow(cost_img, origin='lower', cmap='jet', vmin=0, vmax=0.05)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    test_modes = ['orig', 'rot', 'blur10', 'blur20']
    
    ap = []
    app = []
    apo = []
    pr = []
    
    # evaluate hyperspectral image recovery
    # comment out other SR methods if only want to run one of them
    for mode in test_modes:
        # load file paths of RGB images
        rgb_files = glob('./data/rgb/'+mode+'/*.npy')
        print('#### Now test the', mode, 'images ####')
        
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
            
            # load ground-truth hyperspectral file
            if mode == 'orig':
                gt = load_hyperspectral_data(os.path.join('./data/hyperspectral_gt/orig/', file_name+'.mat'))
            else:
                gt = np.load(os.path.join('./data/hyperspectral_gt/'+mode, file_name+'.npy'))
            
            
            ## A+ Oracle recovery
            print('      A+_Oracle method')
            rec = np.load(os.path.join('./data/hyperspectral_rec/A+_Oracle/'+mode, file_name+'.npy'))
            mrae_eval = mrae(gt, rec)
            print('      ( Mean MRAE: ', np.mean(mrae_eval), ')')
            print('      ( 99pt MRAE: ', np.percentile(mrae_eval, 99), ')')
            show_error_map(mrae_eval, model_name='A+_Oracle')
            apo.append(np.mean(mrae_eval))
            
            
            ## A++ recovery
            print('      A++ method')
            rec = np.load(os.path.join('./data/hyperspectral_rec/A++/'+mode, file_name+'.npy'))
            mrae_eval = mrae(gt, rec)
            print('      ( Mean MRAE: ', np.mean(mrae_eval), ')')
            print('      ( 99pt MRAE: ', np.percentile(mrae_eval, 99), ')')
            show_error_map(mrae_eval, model_name='A++')
            app.append(np.mean(mrae_eval))
            
            
            ## A+ recovery
            print('      A+ method')
            rec = np.load(os.path.join('./data/hyperspectral_rec/A+/'+mode, file_name+'.npy'))
            mrae_eval = mrae(gt, rec)
            print('      ( Mean MRAE: ', np.mean(mrae_eval), ')')
            print('      ( 99pt MRAE: ', np.percentile(mrae_eval, 99), ')')
            show_error_map(mrae_eval, model_name='A+')
            ap.append(np.mean(mrae_eval))
            
            
            ## PR-RELS recovery
            print('      PR-RELS method')
            rec = np.load(os.path.join('./data/hyperspectral_rec/PR-RELS/'+mode, file_name+'.npy'))
            mrae_eval = mrae(gt, rec)
            print('      ( Mean MRAE: ', np.mean(mrae_eval), ')')
            print('      ( 99pt MRAE: ', np.percentile(mrae_eval, 99), ')')
            show_error_map(mrae_eval, model_name='PR-RELS')
            pr.append(np.mean(mrae_eval))
            
            
            ## AWAN recovery
            print('      AWAN method')
            rec = np.load(os.path.join('./data/hyperspectral_rec/AWAN/'+mode, file_name+'.npy'))
            mrae_eval = mrae(gt, rec)
            print('      ( Mean MRAE: ', np.mean(mrae_eval), ')')
            print('      ( 99pt MRAE: ', np.percentile(mrae_eval, 99), ')')
            show_error_map(mrae_eval, model_name='AWAN')
            
            
            ## AWAN-aug3 recovery
            print('      AWAN-aug3 method')
            rec = np.load(os.path.join('./data/hyperspectral_rec/AWAN-aug3/'+mode, file_name+'.npy'))
            mrae_eval = mrae(gt, rec)
            print('      ( Mean MRAE: ', np.mean(mrae_eval), ')')
            print('      ( 99pt MRAE: ', np.percentile(mrae_eval, 99), ')')
            show_error_map(mrae_eval, model_name='AWAN-aug3')
            
            ## HSCNN-D recovery
            print('      HSCNN-D method')
            rec = np.load(os.path.join('./data/hyperspectral_rec/HSCNN-D/'+mode, file_name+'.npy'))
            mrae_eval = mrae(gt, rec)
            print('      ( Mean MRAE: ', np.mean(mrae_eval), ')')
            print('      ( 99pt MRAE: ', np.percentile(mrae_eval, 99), ')')  
            show_error_map(mrae_eval, model_name='HSCNN-D')
            