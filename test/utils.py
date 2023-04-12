# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:05:12 2021
"""

import numpy as np
from scipy.io import loadmat
import utils_sparse as utils_sc
import utils_regression as utils_reg
from utils_sparse import FaissKNeighbors
from utils_regression import RegressionMatrix
    

def load_a_plus_plus_model():
    import pickle5 as pickle
    
    # load K-SVD dictionary and normalize
    spec_rec_anchors_norm = utils_sc.normc(loadmat('./trained_models/dictionary_a_plus_plus.mat')['anchors'].T)
    
    # setup knn
    knn_model = FaissKNeighbors(k=1)
    knn_model.fit(spec_rec_anchors_norm, np.arange(0, spec_rec_anchors_norm.shape[0], 1))
    
    # load trained A++ local regression maps
    path_regmat_a_plus_plus = './trained_models/model_a_plus_plus.pkl'
    with open(path_regmat_a_plus_plus, 'rb') as handle:
        RegMat_a_plus_plus = pickle.load(handle) 

    return knn_model, RegMat_a_plus_plus


def recover_a_plus_plus(rgb, knn_model, RegMat_a_plus_plus):
    ## transform RGB to primary estimate and normalize
    pr_rels_model = load_pr_rels()
    
    primary_spec_rec = recover_pr_rels(rgb, pr_rels_model)
    dim, height, width = primary_spec_rec.shape
    primary_spec_rec = primary_spec_rec.reshape(31, -1).T
    rgb = rgb.reshape(3, -1).T
    
    # normalize the primary spectral estimates
    primary_spec_rec_norm = utils_sc.normc(primary_spec_rec)
    
    ## kNN with adjustable batch size
    nearest_cluster_center = knn_model.predict(primary_spec_rec_norm)
    active_cluster_centers = np.unique(nearest_cluster_center).astype(int)
    
    ## apply local linear maps
    recovery = np.zeros(primary_spec_rec_norm.shape)
    for i in active_cluster_centers:
        is_nearest = nearest_cluster_center == i
        rgb_nearest = rgb[is_nearest, :]
        
        recovery_part = rgb_nearest @ RegMat_a_plus_plus[i].get_matrix()
        recovery[is_nearest, :] = recovery_part
    
    return recovery.T.reshape(31, height, width)


def load_a_plus_model():
    import pickle
    
    # load K-SVD dictionary and normalize
    rgb_anchors_norm = utils_sc.normc(np.load('./trained_models/dictionary_a_plus.npy'))
    
    # setup knn
    knn_model = FaissKNeighbors(k=1)
    knn_model.fit(rgb_anchors_norm, np.arange(0, rgb_anchors_norm.shape[0], 1))
    
    # load trained A+ local regression maps
    path_regmat_a_plus = './trained_models/model_a_plus.pkl'
    with open(path_regmat_a_plus, 'rb') as handle:
        RegMat_a_plus = pickle.load(handle)
    
    return knn_model, RegMat_a_plus
    

def recover_a_plus(rgb, knn_model, RegMat_a_plus):
    # normalize input RGB
    dim, height, width = rgb.shape
    rgb = rgb.reshape(3, -1).T
    rgb_norm = utils_sc.normc(rgb)
    
    # kNN with adjustable batch size
    nearest_cluster_center = knn_model.predict(np.ascontiguousarray(rgb_norm))
    active_cluster_centers = np.unique(nearest_cluster_center).astype(int)
    
    # recovery step
    recovery = np.zeros((rgb_norm.shape[0], 31))
    for i in active_cluster_centers:
        is_nearest = nearest_cluster_center == i
        rgb_nearest = rgb[is_nearest, :]
        
        recovery_part = rgb_nearest @ RegMat_a_plus[i].get_matrix()
        recovery[is_nearest, :] = recovery_part
    
    return recovery.T.reshape(31, height, width)


def load_pr_rels():
    import pickle
    
    path_regmat_pr_rels = './trained_models/model_pr_rels.pkl'
    with open(path_regmat_pr_rels, 'rb') as handle:
        return pickle.load(handle)


def recover_pr_rels(rgb, RegMat_pr_rels):
    # get the polynomial expansion of RGB
    dim, height, width = rgb.shape
    rgb = rgb.reshape(3, -1).T
    poly_rgb = utils_reg.rgb2poly(rgb, 6, root=False)

    # apply regression matrix to polynomial RGB expansions to recover spectra
    recovery = poly_rgb @ RegMat_pr_rels.get_matrix()
    
    return recovery.T.reshape(31, height, width)


def load_AWAN_model(model_type):
    import torch
    from utils_awan.AWAN import AWAN
    
    if model_type == 'orig':
        model_path = './trained_models/model_awan.pth'
    elif model_type == 'aug':
        model_path = './trained_models/model_awan_aug3.pth'
    
    model = AWAN(3, 31, 200, 8)
    save_point = torch.load(model_path, map_location='cpu')#'cuda:0')
    model_param = save_point['state_dict']
    model_dict = {}
    for k1, k2 in zip(model.state_dict(), model_param):
        model_dict[k1] = model_param[k2]
    model.load_state_dict(model_dict)
    
    return model


def recover_AWAN(rgb, model, split_axis):
    from utils_awan.utils import reconstruction_whole_image_cpu
    
    # process input
    dim, height, width = rgb.shape
    rgb = rgb.reshape(3, -1).T
    curr_rgb = (np.float32(rgb).T).reshape(3, height, width)
    curr_rgb = np.transpose(curr_rgb, [0, 2, 1])
    curr_rgb = np.expand_dims(curr_rgb.astype(float), axis=0).copy()
    
    # recovery step
    _, img_res = reconstruction_whole_image_cpu(curr_rgb, model, split_axis)
    
    # post processing recovered data
    img_res = np.swapaxes(np.swapaxes(img_res, 0, 2), 1, 2)
    img_res = np.transpose(img_res, [0, 2, 1])
    recovery = img_res.reshape(31, -1).T
        
    return recovery.T.reshape(31, height, width)


def load_HSCNN_D_model():
    import utils_hscnn_d
    import tensorflow as tf
    
    tf.reset_default_graph()
    
    with tf.device('/cpu:0'):
        lr = tf.placeholder('float', [1, None, None, 3])
	
	# recreate the network
    net = utils_hscnn_d.Net(lr, False, None, 0, reuse=False)
    with tf.device('/cpu:0'):
        net.build_net()
	
	# create a session for running operations in the graph
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
	
	# restore weights
    saver = tf.train.Saver()
    saver.restore(sess, './trained_models/model_hscnn_d.ckpt-113600')
    
    model = {}
    model['lr'] = lr
    model['net'] = net
    model['sess'] = sess
    
    return model


def recover_HSCNN_D(rgb, model, split_axis):
    # extract model
    lr, net, sess = model['lr'], model['net'], model['sess']
    
    # process input
    dim, height, width = rgb.shape
    rgb = rgb.reshape(3, -1).T
    im_lr = np.array(rgb).T.reshape(3,height,width)
    im_lr = np.swapaxes(np.swapaxes(im_lr,0,1),1,2)
    im_lr = np.expand_dims(im_lr, axis=0)
    im_lr = im_lr.astype(np.float32)
    
    # recovery step
    im_sr = np.zeros((im_lr.shape[0],im_lr.shape[1],im_lr.shape[2],31))
    for i in range(16):
        if split_axis == 3:
            temp = sess.run(net.sr, feed_dict={lr: im_lr[:,87*i:87*(i+1),:,:]})
            im_sr[:,87*i:87*(i+1),:,:] = temp
        elif split_axis == 2:
            temp = sess.run(net.sr, feed_dict={lr: im_lr[:,:,87*i:87*(i+1),:]})
            im_sr[:,:,87*i:87*(i+1),:] = temp    
    
    # post processing recovered data
    im_sr = np.squeeze(im_sr)
    im_sr = im_sr.astype(np.float32)
    im_sr = np.swapaxes(np.swapaxes(im_sr,1,2),0,1)
        
    recovery = im_sr.reshape(31, -1).T
    
    return recovery.T.reshape(31, height, width)