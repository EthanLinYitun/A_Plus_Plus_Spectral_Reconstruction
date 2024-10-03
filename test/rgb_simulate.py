# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:27:58 2021
"""

import h5py
import os
import numpy as np
from pandas import read_csv
from glob import glob
from scipy.ndimage import gaussian_filter

from scipy.io import loadmat
def load_hyperspectral_data(file_path):
    '''
    f = h5py.File(file_path,'r')
    for kname, obj in f.items():
        if kname == 'rad':
            output_obj = obj
            break
    
    return np.array(output_obj) / 4095   # size: 31 x 1392 x 1300
    '''
    return np.swapaxes(np.array(loadmat(file_path)['cube']), 0, 2)
    

def interpolate(data, data_waveL, targeted_waveL):
    
    assert data.shape[0] == data_waveL.size, 'Wavelength sequence mismatch with data'
    
    targeted_bounds = [np.min(targeted_waveL), np.max(targeted_waveL)]
    data_bounds = [np.min(data_waveL), np.max(data_waveL)]
    
    assert data_bounds[0] <= targeted_bounds[0], 'targeted wavelength range must be within the original wavelength range'
    assert data_bounds[1] >= targeted_bounds[1], 'targeted wavelength range must be within the original wavelength range'
    
    dim_new_data = list(data.shape)
    dim_new_data[0] = len(targeted_waveL)
    new_data = np.empty(dim_new_data)
    
    for i in range(len(targeted_waveL)):

        relative_L = data_waveL - targeted_waveL[i]
        
        if 0 in relative_L:
            floor = np.argmax( relative_L == 0 )
            new_data[i,...] = data[floor,...]
        
        else:
            floor = np.argmax( relative_L >= 0 ) -1
            interval = data_waveL[floor+1] - data_waveL[floor]
            portion = (targeted_waveL[i] - data_waveL[floor])/interval
            new_data[i,...] = portion*data[floor,...] + (1-portion)*data[floor+1,...]
    
    return new_data 


def load_color_matching_functions(file_path):
    cmf = np.array(read_csv(file_path))[:,1:]         # Color matching functions sampled at every 5 nm. Need interpolation to align with the experimented hyperspectral data
    lambda_cmf = np.array(read_csv(file_path))[:,0]   # (365:830:5)
    
    cmf = interpolate(cmf, lambda_cmf, np.arange(400, 701, 10))   # now the size becomes 31 x 3
    cmf = cmf / np.max(np.sum(cmf, 0))
    
    return cmf


def spec2rgb(spec, cmf):
    dim_spec, height, width = spec.shape
    spec = spec.reshape(dim_spec, -1).T
    rgb = spec @ cmf
    rgb = rgb.T.reshape(3, height, width)
    
    return rgb


def blurring_hyperspectral_image(spec, sigma):
    dim_spec, height, width = spec.shape
    for i in range(dim_spec):
        spec[i, :, :] = gaussian_filter(spec[i, :, :], sigma=sigma)
    
    return spec


if __name__ == '__main__':
    # load ground-truth hyperspectral file paths
    #hyperspectral_files = glob('./data/hyperspectral_gt/orig/*.mat')
    hyperspectral_files = glob('D:\\matR_backup\\ps\\sharpen_gt2\\*.mat')
    
    # load CIE 1964 color matching functions
    cmf = load_color_matching_functions('./resources/cie_1964_cmf.csv')
    
    # generate the RGB images (originals, rotated and blur) and saved as .npy files
    for file_path in hyperspectral_files:
        _, file_name_ext = os.path.split(file_path)
        file_name, _ = os.path.splitext(file_name_ext)
        
        print('Now simulate the RGB image of', file_name)
        
        spec = load_hyperspectral_data(file_path)
        dim_spec, height, width = spec.shape
        
        # save original testing RGB image
        np.save(os.path.join('./data/rgb/orig/', file_name), spec2rgb(spec, cmf))
        
        '''
        # save rotated testing RGB image and processed ground-truth hyperspectral image
        spec_new = np.rot90(spec, k=1, axes=(1,2))
        np.save(os.path.join('./data/hyperspectral_gt/rot/', file_name), spec_new)
        np.save(os.path.join('./data/rgb/rot/', file_name), spec2rgb(spec_new, cmf))
        
        
        # save blur 10 testing RGB image and processed ground-truth hyperspectral image
        spec_new = blurring_hyperspectral_image(spec, 10)
        np.save(os.path.join('./data/hyperspectral_gt/blur10/', file_name), spec_new)
        np.save(os.path.join('./data/rgb/blur10/', file_name), spec2rgb(spec_new, cmf))
        
        # save blur 20 testing RGB image and processed ground-truth hyperspectral image
        spec_new = blurring_hyperspectral_image(spec, 20)
        np.save(os.path.join('./data/hyperspectral_gt/blur20/', file_name), spec_new)
        np.save(os.path.join('./data/rgb/blur20/', file_name), spec2rgb(spec_new, cmf))
        '''