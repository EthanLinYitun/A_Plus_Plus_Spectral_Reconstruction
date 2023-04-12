# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:51:48 2021
"""

import numpy as np
import os
import utils
import utils_regression as utils_reg
from utils_regression import RegressionMatrix
import utils_sparse as utils_sc
from utils_sparse import FaissKNeighbors
from rgb_simulate import load_hyperspectral_data, load_color_matching_functions
from scipy.io import loadmat
import pickle
from scipy.spatial.distance import cdist
from evaluate import mrae


def down_sampling_training_data(im_list):
    cmf = load_color_matching_functions('./resources/cie_1964_cmf.csv')
    data_dir = 'E:/UEA/Resource/ICVL/'
    
    gt_data = {'spec': [],
               'rgb': []}
    
    for im_name in im_list:
        print('     ', im_name[:-1])
        spec_img = load_hyperspectral_data(os.path.join(data_dir, im_name[:-1])) # 31 x H x W
        
        spec_data = spec_img.reshape(spec_img.shape[0], -1).T # Dim_Data x 31
        spec_data = utils_reg.sampling_data(spec_data, num_sampling_points=30000, rand=False)     
    
        rgb_data = spec_data @ cmf
        
        gt_data['spec'] = gt_data['spec'] + [spec_data]
        gt_data['rgb']  = gt_data['rgb']  + [rgb_data]
    
    gt_data['spec'] = np.array(gt_data['spec']).swapaxes(0, 2).reshape(spec_data.shape[1], -1).T
    gt_data['rgb']  = np.array(gt_data['rgb']).swapaxes(0, 2).reshape(rgb_data.shape[1], -1).T
    
    return gt_data


def knn(data, reference, k, batch_size=None):
    num_reference = reference.shape[0]
    if batch_size:
        pass
    else:
        batch_size = num_reference
    num_batch = num_reference//batch_size
    num_residual = num_reference%batch_size
    out_list = np.zeros((num_reference, k))
    for i in range(num_batch):
        D = cdist(reference[i*batch_size:(i+1)*batch_size, :], data, "sqeuclidean")
        out_list[i*batch_size:(i+1)*batch_size, :] = np.argsort(D, axis=1)[:, :k]
    if num_residual:
        D = cdist(reference[-num_residual:, :], data, "sqeuclidean")
        out_list[-num_residual:, :] = np.argsort(D, axis=1)[:, :k]
    
    return out_list


def train(im_list):
    # down sample the training data
    gt_data = down_sampling_training_data(im_list)
    
    # calculate primary estimates and normalize
    path_regmat_pr_rels = './resources/model_pr_rels.pkl'
    with open(path_regmat_pr_rels, 'rb') as handle:
        model_pr_rels = pickle.load(handle)
    primary_estimates = utils.recover_pr_rels(gt_data['rgb'].T.reshape(3, 1, -1), model_pr_rels)
    primary_estimates = primary_estimates.reshape(31, -1).T
    primary_estimates_norm = utils_sc.normc(primary_estimates)
        
    # load K-SVD dictionary and normalize
    primary_estimates_cluster_centers_norm = utils_sc.normc(loadmat('./resources/dictionary_a_plus_plus.mat')['anchors'].T)
    
    # calculate the nearest neighbors of each cluster center
    nearest_neighbors = knn(primary_estimates_norm, primary_estimates_cluster_centers_norm, k=8192, batch_size=250).astype(int)
    num_centers, num_neighbors = nearest_neighbors.shape
    
    # Training "Multiple" Regression Matrix
    RegMat = []
    regress_mode = {'type': 'poly', 
                    'order': 1,
                    'dim_spec': 31,
                    'dim_rgb': 3,
                    'num_anchors': 8192,
                    'target_wavelength': np.arange(400,701,10)}
    
    advanced_mode = {'Rel_Fit': False,
                     'Sparse': True}
    
    for i in range(num_centers):
        RegMat.append(utils_reg.RegressionMatrix(regress_mode, advanced_mode))
        
    for i in range(num_centers):
        nearest_idx = nearest_neighbors[i, :]
        gt_data_nearest = {}
        gt_data_nearest['spec'] = gt_data['spec'][nearest_idx, :]
        gt_data_nearest['rgb'] = gt_data['rgb'][nearest_idx, :]
        
        RegMat[i].update(gt_data_nearest['rgb'], gt_data_nearest['spec'])
    
    with open(os.path.join('./trained_models/', 'model_a_plus_plus.pkl'), 'wb') as handle:
        pickle.dump(RegMat, handle, protocol=pickle.HIGHEST_PROTOCOL)

#im_list = open('E:/UEA/Code/Sparse_Oracle/resources/fn_icvl_group_A.txt').readlines()
#train(im_list[:3])


def validate(im_list):    
    # down sample the validation data
    gt_data = down_sampling_training_data(im_list)
    
    # calculate primary estimates and normalize
    path_regmat_pr_rels = './resources/model_pr_rels.pkl'
    with open(path_regmat_pr_rels, 'rb') as handle:
        model_pr_rels = pickle.load(handle)
    primary_estimates = utils.recover_pr_rels(gt_data['rgb'].T.reshape(3, 1, -1), model_pr_rels)
    primary_estimates = primary_estimates.reshape(31, -1).T
    primary_estimates_norm = utils_sc.normc(primary_estimates)
    
    # load K-SVD dictionary and normalize
    primary_estimates_cluster_centers_norm = utils_sc.normc(loadmat('./resources/dictionary_a_plus_plus.mat')['anchors'].T)
    
    # calculate the nearest neighbors of each cluster center
    nearest_neighbors = knn(primary_estimates_norm, primary_estimates_cluster_centers_norm, k=8192//2, batch_size=250).astype(int)
    num_centers, num_neighbors = nearest_neighbors.shape
    
    regress_mode = {'type': 'poly', 
                    'order': 1,
                    'dim_spec': 31,
                    'dim_rgb': 3,
                    'num_anchors': 8192,
                    'target_wavelength': np.arange(400,701,10)}
    
    advanced_mode = {'Rel_Fit': False,
                     'Sparse': True}
    
    with open(os.path.join('./trained_models/', 'model_a_plus_plus.pkl'), 'rb') as handle:
        RegMat = pickle.load(handle)
    
    # Regularizing Regression Matrix
    for i in range(num_centers):
        if i%100 == 50:
            print('    anchor', i)
        nearest_idx_val = nearest_neighbors[i, :]
        gt_data_nearest_val = {}
        gt_data_nearest_val['spec'] = gt_data['spec'][nearest_idx_val, :]
        gt_data_nearest_val['rgb'] = gt_data['rgb'][nearest_idx_val, :]
        
        RegMat[i] = utils_reg.regularize(RegMat[i], gt_data_nearest_val['rgb'], gt_data_nearest_val['spec'], advanced_mode, 
                                         mrae, show_graph=False)
    
    with open(os.path.join('./trained_models/', 'model_a_plus_plus.pkl'), 'wb') as handle:
        pickle.dump(RegMat, handle, protocol=pickle.HIGHEST_PROTOCOL)

im_list = open('E:/UEA/Code/Sparse_Oracle/resources/fn_icvl_group_D.txt').readlines()
validate(im_list[:3])