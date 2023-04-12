# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 09:46:42 2021

@author: vby18pwu
"""

import tensorflow as tf
import numpy as np

import os
import scipy.io as sio

def _atrous_conv2d(value, filters, rate, padding, name=None):
	return tf.nn.convolution(
		input=value,
		filter=filters,
		padding=padding,
		dilation_rate=np.broadcast_to(rate, (2,)),
	    data_format='NHWC',
	    name=name)


def conv2d(inputs, num_outputs, kernel_shape=[3, 3], strides=[1, 1], add_biases=True, pad='SAME', dilated=1, reuse=False, tower_index=None,
           W_init=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False), # msra
           b_init=tf.constant_initializer(0.0), W_params=None, b_params=None, wl=None, wl_type=tf.nn.l2_loss, summary=False, scope='conv2d'):
	"""
	Args:
	  inputs: NHWC
	  num_outputs: the number of filters
	  kernel_shape: [height, width]
	  strides: [height, width]
	  pad: 'SAME' or 'VALID'
	  W/b_params: lists for layer-wise learning rate and gradient clipping
	  wl: add weight losses to collection
	  reuse: reusage of variables
	  dilated: convolution with holes
	
	Returns:
	  outputs: NHWC
	"""
	with tf.variable_scope(scope, reuse=reuse):
		# get shapes
		kernel_h, kernel_w = kernel_shape
		stride_h, stride_w = strides
		batch_size, height, width, in_channel = inputs.get_shape().as_list()
		
		weights_shape = [kernel_h, kernel_w, in_channel, num_outputs]
		weights = tf.get_variable('w', weights_shape, tf.float32, W_init)
		
		# add summary for w
		if summary and not reuse:
			tf.summary.histogram('hist_w', weights)
		
		# add to the list of weights
		if W_params is not None and not reuse:
			W_params += [weights]
		
		# 2-D convolution
		if dilated != 1:
			outputs = _atrous_conv2d(inputs, weights, rate=dilated, padding=pad)
		else:
			outputs = tf.nn.conv2d(inputs, weights, [1, stride_h, stride_w, 1], padding=pad, data_format='NHWC')
		
		# add biases
		if add_biases:
			biases = tf.get_variable('b', [num_outputs], tf.float32, b_init)
			
			# add summary for b
			if summary and not reuse:
				tf.summary.histogram('hist_b', biases)
				
			# add to the list of biases
			if b_params is not None and not reuse:
				b_params += [biases]
			
			outputs = tf.nn.bias_add(outputs, biases, data_format='NHWC')
		
		# add weight decay
		if wl is not None:
			weight_loss = tf.multiply(wl_type(weights), wl, name='weight_loss')
			tf.add_to_collection('losses_' + str(tower_index), weight_loss)
		
		return outputs

def modcrop(im, modulo):
	if len(im.shape) == 3:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1], :]
	elif len(im.shape) == 2:
		size = np.array(im.shape)
		size = size - (size % modulo)
		im = im[0 : size[0], 0 : size[1]]
	else:
		raise AttributeError
	return im


def shave(im, border):
	if len(im.shape) == 3:
		return im[border[0] : -border[0], 
		          border[1] : -border[1], :]
	elif len(im.shape) == 2:
		return im[border[0] : -border[0], 
		          border[1] : -border[1]]
	else:
		raise AttributeError


class Net(object):
	def __init__(self, lr, non_local, wl, tower, reuse):
		# training inputset
		self.lr = lr
		
		# multi-gpu related settings
		self.reuse = reuse
		self.tower = tower
		
		# parameter lists for weights and biases
		self.W_params = []
		self.b_params = []
		
		# coefficient of weight decay
		self.wl = wl
		
		# whether to enable the non-local block
		self.non_local = non_local
	
	
	def dfus_block(self, bottom, i):
		act = tf.nn.relu

		with tf.variable_scope('dfus_block' + str(i), reuse=self.reuse):
			conv1  = act(conv2d(bottom, 64, [1, 1], wl=None, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_i'), name='relu' + str(i) + '_i')

			feat1  = act(conv2d(conv1, 16, [3, 3], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_1'), name='relu' + str(i) + '_1')
			feat15 = act(conv2d(feat1, 8, [1, 1], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_15'), name='relu' + str(i) + '_15')

			feat2  = act(conv2d(conv1, 16, [3, 3], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_2'), name='relu' + str(i) + '_2')
			feat23 = act(conv2d(feat2, 8, [1, 1], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_23'), name='relu' + str(i) + '_23')

			feat = tf.concat([feat1, feat15, feat2, feat23], 3, name='conv' + str(i) + '_c1')
			feat = act(conv2d(feat, 16, [1, 1], wl=None, reuse=self.reuse, tower_index=self.tower, scope='conv' + str(i) + '_r'), name='relu' + str(i) + '_r')

			top = tf.concat([bottom, feat], 3, name='conv' + str(i) + '_c2')

		return top


	def ddfn(self, bottom, step, b=10):
		act = tf.nn.relu

		with tf.variable_scope('ddfn_' + str(step), reuse=self.reuse):
			with tf.name_scope('msfeat'):
				conv13  = act(conv2d(bottom, 16, [3, 3], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv1_3'), name='relu1_3')
				conv15  = act(conv2d(bottom, 16, [3, 3], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv1_5'), name='relu1_5')

				conv135 = act(conv2d(conv13, 16, [1, 1], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv1_3_5'), name='relu1_3_5')
				conv153 = act(conv2d(conv15, 16, [1, 1], wl=self.wl, reuse=self.reuse, tower_index=self.tower, scope='conv1_5_3'), name='relu1_5_3')

				conv1 = tf.concat([conv13, conv15, conv135, conv153], 3, name='conv1_c')

			if self.non_local:
				conv1, _ = non_local_block(conv1, reuse=self.reuse, tower_index=self.tower)

			feat = self.dfus_block(conv1, 2)

			for i in range(3, b, 1):
				feat = self.dfus_block(feat, i)

			top = feat

			return top
	
	
	def build_net(self):
		with tf.variable_scope('net', reuse=self.reuse):
			feat0 = self.ddfn(self.lr, 0, b=60)
			feat1 = self.ddfn(self.lr, 1, b=60)
			feat2 = self.ddfn(self.lr, 2, b=60)
			feat = tf.concat([feat0, feat1, feat2], axis=3)
			
			outputs = conv2d(feat, 31, [1, 1], W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.001),
			                add_biases=True, wl=None, reuse=self.reuse, tower_index=self.tower, scope='fusion')
			self.sr = outputs

