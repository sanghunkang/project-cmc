#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

img = Image.open("../../data_light/bmp/I0000001.BMP")
img = Image.open("..\\..\\data_light/bmp/I0000001.BMP")
img = img.resize((224, 224))
arr_img = np.asarray(img, dtype=np.float32)
print(arr_img.shape)
print(arr_img.dtype)

DIR_DATA_WEIGHTPRETRAINED = "C:\\dev-data\\weightPretrained\\"
# DIR_DATA_WEIGHTPRETRAINED = "../../../../dev-data/weightPretrained/"

# Define some functions... for whatever purposes
def reformat_params(dict_lyr):
	params_pre = {}
	for key in dict_lyr:
		params_pre[key + "_W"] = tf.Variable(dict_lyr[key]["weights"], name=key + "_W")
		params_pre[key + "_b"] = tf.Variable(dict_lyr[key]["biases"], name=key + "_b")
	return params_pre

def slice_params_module(name_module, params_pre):
	params_module = {}
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

def inception_module(tsr_X, params_module):
	# 1x1 convolution
	inception_4b_1x1 = conv2d(tsr_X, params_module["1x1_W"], params_module["1x1_b"])

	# 3x3 convolution
	inception_4b_3x3r = conv2d(tsr_X, params_module["3x3_reduce_W"], params_module["3x3_reduce_b"])
	inception_4b_3x3 = conv2d(inception_4b_3x3r, params_module["3x3_W"], params_module["3x3_b"])

	# 5x5 convolution
	inception_4b_5x5r = conv2d(tsr_X, params_module["5x5_reduce_W"], params_module["5x5_reduce_b"])
	inception_4b_5x5 = conv2d(inception_4b_5x5r, params_module["5x5_W"], params_module["5x5_b"])

	# 3x3 max-pool
	inception_4b_pool_proj = conv2d(tsr_X, params_module["pool_proj_W"], params_module["pool_proj_b"])

	# Concatenation of sublayers - i.e. stacking
	inception_4b_concat = tf.concat([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj], axis=-1)
	return inception_4b_concat

# GoogLeNet
print("++++++++++ GoogLeNet ++++++++++")
dict_lyr = np.load(DIR_DATA_WEIGHTPRETRAINED + "googlenet.npy", encoding = 'latin1').item() # return dict

params_pre = reformat_params(dict_lyr)

keys = []
for x in params_pre: keys.append(x)

keys.sort()
for key in keys: print(key, params_pre[key].get_shape())

# Model
tsr_X = tf.reshape(arr_img, shape=[-1, 224, 224, 3])
conv1_7x7_s2 = conv2d(tsr_X, params_pre['conv1_7x7_s2_W'], params_pre['conv1_7x7_s2_b'], strides=2)
conv1_7x7p_s2 = maxpool2d(conv1_7x7_s2, k=2)
conv2_3x3 = conv2d(conv1_7x7p_s2, params_pre['conv2_3x3_W'], params_pre['conv2_3x3_b'])
conv2p_3x3 = maxpool2d(conv2_3x3, k=2)

# Slice into subsets
params_inception_3a = slice_params_module("inception_3a_", params_pre)
params_inception_3b = slice_params_module("inception_3b_", params_pre)
params_inception_4a = slice_params_module("inception_4a_", params_pre)
params_inception_4b = slice_params_module("inception_4b_", params_pre)
params_inception_4c = slice_params_module("inception_4c_", params_pre)
params_inception_4d = slice_params_module("inception_4d_", params_pre)
params_inception_4e = slice_params_module("inception_4e_", params_pre)
params_inception_5a = slice_params_module("inception_5a_", params_pre)
params_inception_5b = slice_params_module("inception_5b_", params_pre)

# Convolution and max pooling(down-sampling) Layers
# Parameters are from pretrained data
inception_3a = inception_module(conv2p_3x3, params_inception_3a)
inception_3b = inception_module(inception_3a, params_inception_3b)
inception_3bp = maxpool2d(inception_3b, k=2)

inception_4a = inception_module(inception_3bp, params_inception_4a)
inception_4b = inception_module(inception_4a, params_inception_4b)
inception_4c = inception_module(inception_4b, params_inception_4c)
inception_4d = inception_module(inception_4c, params_inception_4d)
inception_4e = inception_module(inception_4d, params_inception_4e)
inception_4ep = maxpool2d(inception_4e, k=2)

inception_5a = inception_module(inception_4ep, params_inception_5a)
inception_5b = inception_module(inception_5a, params_inception_5b)
inception_5ap = tf.nn.avg_pool(inception_5b, ksize=[1, 1, 1, 1], strides=[1, 7, 7, 1], padding='SAME')
print(inception_4ep.get_shape())
print(inception_5a.get_shape())
print(inception_5b.get_shape())
print(inception_5ap.get_shape())

# Fully connected layer (and Apply Dropout)
# Training (or tuning) is done here

inception_fc = tf.reshape(inception_5ap, [-1, params_pre["loss3_classifier_W"].get_shape().as_list()[0]])
print(inception_fc.get_shape())
linclassifier = tf.add(tf.matmul(inception_fc, params_pre["loss3_classifier_W"]), params_pre["loss3_classifier_b"])
linclassifier = tf.nn.relu(linclassifier)
# fc1 = tf.nn.dropout(fc1, params['dropout'])

with tf.Session() as sess:
	# Initialise the variables and run
	init = tf.global_variables_initializer()
	sess.run(init)
	
	# with tf.device("/cpu:0"):
	with tf.device("/gpu:0"):
		linclassifier_eval = linclassifier.eval()[0]
		print(np.argmax(linclassifier_eval))

		conv23_eval = conv2_3x3.eval()[0]
		conv23_eval = np.swapaxes(conv23_eval, 0, 2) # (16(x),16(y),64)
		conv23_eval = np.swapaxes(conv23_eval, 1, 2)

		# for rec in conv23_eval[20:24]:
		# 	plt.imshow(rec, cmap="gray")
		# 	plt.show()
