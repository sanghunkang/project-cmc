#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages

# Import external packages
import numpy as np
import tensorflow as tf

def slice_params_module(name_module, params_pre):
	params_module = {}
	keys = [key for key in params_pre]
	for key in keys:
		if name_module in key:
			params_module[key.replace(name_module,"")] = params_pre[key]
	return params_module

def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv2ser(x, params_fc):
	shape_ftmap_end = x.get_shape()
	len_ftmap_end = int(shape_ftmap_end[1]*shape_ftmap_end[2]*shape_ftmap_end[3])
	return tf.reshape(x, [-1, params_fc.get_shape().as_list()[0]])

def fc1d(x, W, b,  bn=False):
	# FC layer wrapper, with bias, relu activation plus batch-normalisation if demanded
	fc = tf.add(tf.matmul(x, W), b)
	if bn == True: fc = tf.contrib.layers.batch_norm(fc)
	return tf.nn.relu(fc)
	
def inception_module(tsr_X, name_module, params_pre):
	params_module = slice_params_module(name_module, params_pre)
	# 1x1 convolution
	inception_1x1 = conv2d(tsr_X, params_module["1x1_W"], params_module["1x1_b"])

	# 3x3 convolution
	inception_3x3r = conv2d(tsr_X, params_module["3x3_reduce_W"], params_module["3x3_reduce_b"])
	inception_3x3 = conv2d(inception_3x3r, params_module["3x3_W"], params_module["3x3_b"])

	# 5x5 convolution
	inception_5x5r = conv2d(tsr_X, params_module["5x5_reduce_W"], params_module["5x5_reduce_b"])
	inception_5x5 = conv2d(inception_5x5r, params_module["5x5_W"], params_module["5x5_b"])

	# 3x3 max-pool
	inception_pool_proj = conv2d(tsr_X, params_module["pool_proj_W"], params_module["pool_proj_b"])

	# Concatenation of sublayers - i.e. stacking
	inception_concat = tf.concat([inception_1x1, inception_3x3, inception_5x5, inception_pool_proj], axis=-1)
	return inception_concat

def arxtect_inceptionv1(X, params_pre, params):
	X_reshaped = tf.reshape(X, shape=[-1, 448, 448, 3])

	# Convolution and max pooling(down-sampling) Layers
	# Convolution parameters are from pretrained data
	conv1_7x7_s2 = conv2d(X_reshaped, params_pre['conv1_7x7_s2_W'], params_pre['conv1_7x7_s2_b'], strides=2)
	conv1_7x7p_s2 = maxpool2d(conv1_7x7_s2, k=2)

	conv2_3x3 = conv2d(conv1_7x7p_s2, params_pre['conv2_3x3_W'], params_pre['conv2_3x3_b'])
	conv2p_3x3 = maxpool2d(conv2_3x3, k=2)

	# Inception modules
	inception_3a = inception_module(conv2p_3x3, "inception_3a_", params_pre)
	inception_3b = inception_module(inception_3a, "inception_3b_", params_pre)
	inception_3bp = maxpool2d(inception_3b, k=2)

	inception_4a = inception_module(inception_3bp, "inception_4a_", params_pre)
	inception_4b = inception_module(inception_4a, "inception_4b_", params_pre)
	inception_4c = inception_module(inception_4b, "inception_4c_", params_pre)
	inception_4d = inception_module(inception_4c, "inception_4d_", params_pre)
	inception_4e = inception_module(inception_4d, "inception_4e_", params_pre)
	inception_4ep = maxpool2d(inception_4e, k=2)

	inception_5a = inception_module(inception_4ep, "inception_5a_", params_pre)
	inception_5b = inception_module(inception_5a, "inception_5b_", params_pre)
	print(inception_5b.get_shape())
	inception_5ap = tf.nn.avg_pool(inception_5b, ksize=[1, 1, 1, 1], strides=[1, 7, 7, 1], padding='SAME')
	print(inception_5ap.get_shape())

	# Fully connected layer, training is done only for here
	inception_5ap_1d = conv2ser(inception_5ap, params["fc6_W"])
	fc6 = fc1d(inception_5ap_1d, params["fc6_W"], params["fc6_b"],  bn=True)
	fc7 = fc1d(fc6, params["fc7_W"], params["fc7_b"], bn=True)
	pred = fc1d(fc7, params["fc8_W"], params["fc8_b"], bn=True)
	return pred
