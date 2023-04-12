# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:33:33 2021
"""

import numpy as np
from itertools import combinations_with_replacement
from numpy.linalg import inv, det
import matplotlib.pyplot as plt

def get_polynomial_terms(num_of_var, highest_order, root):
    if highest_order == 1:
        all_set = np.eye(num_of_var)
        #final_set = [(1,0,0),(0,1,0),(0,0,1)]
        final_set = [tuple(all_set[i, :]) for i in range(num_of_var)]
        
        return final_set
    
    final_set = set()   # save the set of polynomial terms
    index_of_variables = [i for i in range(num_of_var)]
    
    for order in range(1,highest_order+1):  # consider all higher order terms from order 1, excluding the constant term
        
        # Each list member: one composition of the term of the assigned order, in terms of variable indices      
        curr_polynomial_terms = list(combinations_with_replacement(index_of_variables,order))   
        
        for t in range(len(curr_polynomial_terms)):
            curr_term = curr_polynomial_terms[t]
            mapped_term = np.zeros(num_of_var)       # save the index value of each variables
            
            for var in curr_term:
                if root:
                    mapped_term[var] += 1./order
                else:
                    mapped_term[var] += 1.
                    
            final_set.add(tuple(mapped_term))
        
    return list(sorted(final_set))


def rgb2poly(rgb_data, poly_order, root):
    dim_data, dim_variables = rgb_data.shape
    poly_term = get_polynomial_terms(dim_variables, poly_order, root)
    dim_poly = len(poly_term)
    
    out_mat = np.empty((dim_data, dim_poly))
    
    for term in range(dim_poly):
        new_col = np.ones((dim_data))            # DIM_DATA,
        for var in range(dim_variables):
            variable_vector = rgb_data[:, var]                             # DIM_DATA,
            variable_index_value = poly_term[term][var]
            new_col = new_col * ( variable_vector**variable_index_value )
            
        out_mat[:,term] = new_col
    
    return out_mat


def get_regression_parts(data_spec, data_from_rgb, weights=()):
    '''
    Input data_spec with shape ( DIM_DATA, DIM_SPEC )
          data_from_rgb with shape ( DIM_DATA, -1 ), could be data_poly or data_patch
    Output squared_term, body_term
    '''
    
    if weights == ():
        squared_term = data_from_rgb.T @ data_from_rgb    # DIM_RGB x DIM_RGB
        body_term = data_from_rgb.T @ data_spec      # DIM_RGB x DIM_SPEC
    else:
        weights = weights.reshape(1, -1)
        
        squared_term = (data_from_rgb.T * weights) @ data_from_rgb    # DIM_RGB x DIM_RGB
        body_term = (data_from_rgb.T * weights) @ data_spec      # DIM_RGB x DIM_SPEC
    
    return squared_term, body_term


class RegressionMatrix():
    def __init__(self, regress_mode, advanced_mode):
        
        self.regress_mode = regress_mode
        self.advanced_mode = advanced_mode
        
        if regress_mode['type'] == 'poly':
            self.__dim_regress_input = len(get_polynomial_terms(regress_mode['dim_rgb'], regress_mode['order'], False))
        
        self.__dim_regress_output = regress_mode['dim_spec']
        
        if advanced_mode['Rel_Fit']:
            self.__squared_term = [np.zeros((self.__dim_regress_input, self.__dim_regress_input))] * self.__dim_regress_output
            self.__body_term = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__matrix = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__gamma_ch = np.zeros(self.__dim_regress_output)
        else:
            self.__squared_term = np.zeros((self.__dim_regress_input, self.__dim_regress_input))
            self.__body_term = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__matrix = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
            self.__gamma = 1
            
    def set_gamma(self, gamma, channel=(), regress_input_tr=(), regress_output_tr=(), weights_sqr=(), weights_reg=()):
        if self.advanced_mode['Rel_Fit']:
            self.__gamma_ch[channel] = gamma
            self.__matrix[:, channel] = inv( self.__squared_term[channel] + self.__gamma_ch[channel] * np.eye(self.__dim_regress_input) ) @ self.__body_term[:, channel]
        else:
            self.__gamma = gamma
            self.__matrix = inv( self.__squared_term + self.__gamma * np.eye(self.__dim_regress_input) ) @ self.__body_term
    
    def get_gamma(self, channel=()):
        if self.advanced_mode['Rel_Fit']:   
            return self.__gamma_ch[channel]
        else:
            return self.__gamma
    
    def get_matrix(self):
        return self.__matrix
    
    def get_dim_regress_input(self):
        return self.__dim_regress_input
    
    def get_dim_regress_output(self):
        return self.__dim_regress_output
    
    def reset_weights(self):
        self.__weights_reg = ()
        self.__weights_sqr = ()
        self.__num_data = ()
    
    def test_feasible_gamma(self, gamma, channel=()):
        if self.advanced_mode['Rel_Fit']:
            return det(self.__squared_term[channel] + gamma * np.eye(self.__dim_regress_input)) != 0
        else:
            return det(self.__squared_term + gamma * np.eye(self.__dim_regress_input)) != 0
    
    def update(self, regress_input, regress_output):
        if self.advanced_mode['Rel_Fit']:
            num_data = regress_input.shape[0]
            for channel in range(self.__dim_regress_output):
                squared_term, body_term = get_regression_parts(np.ones(num_data), 
                                                               1./regress_output[:, channel].reshape(num_data, 1) * regress_input)            
                self.__squared_term[channel] = self.__squared_term[channel] + squared_term
                self.__body_term[:, channel] = self.__body_term[:, channel] + body_term
        else:
            squared_term, body_term = get_regression_parts(regress_output, regress_input)
            self.__squared_term = self.__squared_term + squared_term
            self.__body_term = self.__body_term + body_term
    

def sampling_data(data, num_sampling_points, rand=False):
    if rand:
        np.random.shuffle(data)
        
    sampling_points = np.floor(np.linspace(0, len(data), num_sampling_points, endpoint=False)).astype(int)
    return data[sampling_points, :]


def recover(regress_matrix, regress_input, advanced_mode, resources, gt_rgb=(), exposure=1):
    
    recovery = {}
    recovery['spec'] = regress_input @ regress_matrix
    recovery['rgb'] = recovery['spec'] @ resources['cmf']
    
    return recovery


def per_channel_recover(regress_matrix, channel, regress_input, advanced_mode, resources, gt_data=(), exposure=1):
    
    recovery_ch = {}
    gt_data_ch = {}
    
    recovery_ch['spec'] = regress_input @ regress_matrix[:, channel] # DIM_Data,
    
    if len(gt_data) == 0:
        gt_data_ch = ()
    else:
        gt_data_ch['spec'] = gt_data['spec'][:, channel].reshape(-1, 1)
    recovery_ch['spec'] = recovery_ch['spec'].reshape(-1, 1)
    
    return gt_data_ch, recovery_ch 


def regularize(RegMat, regress_input, gt_data, advanced_mode, cost_func, resources=(), regress_input_tr=(), regress_output_tr=(), regress_mode=(), show_graph=False):
    
    def determine_feasible_gamma(channel=(), max_range=None):
        if max_range:
            pass
        else:
            max_range = 20
            
        for s in range(-max_range, 0, 1):
            if RegMat.test_feasible_gamma(10**s, channel):
                break
        #print('feasible range:', s, 'to', 10)
        return np.logspace(10, s, np.abs(10-s))
    
    def regularizer(test_gammas, channel=(), tolerance = 0.00005, return_best_model=False):
        cost = []
        best_weights_sqr = []
        best_weights_reg = []
        
        for gamma in test_gammas:
            if advanced_mode['Rel_Fit']:
                RegMat.set_gamma(gamma, channel)
                gt_data_ch, recovery_ch = per_channel_recover(RegMat.get_matrix(), channel, regress_input, advanced_mode, resources, gt_data)
                cost.append(np.mean(cost_func(gt_data_ch, recovery_ch)))
            
            else:
                RegMat.set_gamma(gamma)        
                recovery = recover(RegMat.get_matrix(), regress_input, advanced_mode, resources, gt_data['rgb'])  
                cost.append(np.mean(cost_func(gt_data, recovery)))
            
        best_gamma = test_gammas[np.argmin(cost)]
        
        if show_graph:
            plt.figure()
            plt.title('Tikhonov parameter search')
            plt.plot(test_gammas, cost)
            plt.scatter(best_gamma, np.min(cost), c='r', marker='o')
            plt.xscale('log')
            plt.show()
        
        if return_best_model:
            return best_gamma, best_weights_sqr[np.argmin(cost)], best_weights_reg[np.argmin(cost)]
        else:
            return best_gamma
    
    if advanced_mode['Rel_Fit']:
        for channel in range(RegMat.get_dim_regress_output()):
            test_gammas = determine_feasible_gamma(channel)
            best_gamma = regularizer(test_gammas, channel)
            test_gammas_fine = best_gamma * np.logspace(-1, 1, 1000)
            best_gamma = regularizer(test_gammas_fine, channel)            
            RegMat.set_gamma(best_gamma, channel)    
    else:
        test_gammas = determine_feasible_gamma()
        best_gamma = regularizer(test_gammas)
        test_gammas_fine = best_gamma * np.logspace(-1, 1, 1000)
        best_gamma = regularizer(test_gammas_fine)
        RegMat.set_gamma(best_gamma)
    
    return RegMat