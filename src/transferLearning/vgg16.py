#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# sess = tf.Session()
# imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

DIR_DATA_WEIGHTPRETRAINED = "C:\\dev-data\\weightPretrained\\"
# DIR_DATA_WEIGHTPRETRAINED = "../../../../dev-data/weightPretrained/"

# VGGNet
print("++++++++++ VGGNet ++++++++++")
# aa = np.load("pretrained\\vgg16.npz")

def reformat_params(dict_lyr):
	params_pre = {}
	for key in dict_lyr:
		params_pre[key + "_W"] = tf.Variable(dict_lyr[key][0], name=key + "_W")
		params_pre[key + "_b"] = tf.Variable(dict_lyr[key][1], name=key + "_b")
	return params_pre

dict_lyr = np.load(DIR_DATA_WEIGHTPRETRAINED + "vgg16.npy", encoding = "latin1").item() # return dict
params_pre = reformat_params(dict_lyr)

keys = []
for x in params_pre: keys.append(x)

keys.sort()


# GoogLeNet
# print("++++++++++ GoogLeNet ++++++++++")
# aa = np.load("pretrained\\googlenet.npy", encoding = 'latin1').item() # return dict
# for a in aa: print(a, aa[a]["biases"].shape, aa[a]["weights"].shape)

import PIL.Image as Image

img = Image.open("../../data_light/bmp/I0000001.BMP")
img = Image.open("..\\..\\data_light/bmp/I0000001.BMP")
img = img.resize((224,224))
arr_img = np.asarray(img, dtype=np.float32)
print(arr_img.shape)
print(arr_img.dtype)

for key in keys: print(key, params_pre[key])


def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# X = tf.convert_to_tensor(arr_img, dtype=tf.float32)
tsr_X = tf.reshape(arr_img, shape=[-1, 224, 224, 3])

# Convolution and max pooling(down-sampling) Layers
# Parameters are from pretrained data
conv11 = conv2d(tsr_X, params_pre['conv1_1_W'], params_pre['conv1_1_b'])
conv12 = conv2d(conv11, params_pre['conv1_2_W'], params_pre['conv1_2_b'])
conv1p = maxpool2d(conv12, k=2)

conv21 = conv2d(conv1p, params_pre['conv2_1_W'], params_pre['conv2_1_b'])
conv22 = conv2d(conv21, params_pre['conv2_2_W'], params_pre['conv2_2_b'])
conv2p = maxpool2d(conv22, k=2)

conv31 = conv2d(conv2p, params_pre['conv3_1_W'], params_pre['conv3_1_b'])
conv32 = conv2d(conv31, params_pre['conv3_2_W'], params_pre['conv3_2_b'])
conv3p = maxpool2d(conv32, k=2)

conv41 = conv2d(conv3p, params_pre['conv4_1_W'], params_pre['conv4_1_b'])
conv42 = conv2d(conv41, params_pre['conv4_2_W'], params_pre['conv4_2_b'])
conv4p = maxpool2d(conv42, k=2)

conv51 = conv2d(conv4p, params_pre['conv5_1_W'], params_pre['conv5_1_b'])
conv52 = conv2d(conv51, params_pre['conv5_2_W'], params_pre['conv5_2_b'])
conv5p = maxpool2d(conv52, k=2)

fc6 = tf.reshape(conv5p, [-1, params_pre["fc6_W"].get_shape().as_list()[0]])

# Fully connected layer (and Apply Dropout)
# Training is done here
fc6 = tf.add(tf.matmul(fc6, params_pre["fc6_W"]), params_pre["fc6_b"])
fc6 = tf.nn.relu(fc6)
# fc1 = tf.nn.dropout(fc1, params['dropout'])

fc7 = tf.add(tf.matmul(fc6, params_pre["fc7_W"]), params_pre["fc7_b"])
fc7 = tf.nn.relu(fc7)

fc8 = tf.add(tf.matmul(fc7, params_pre["fc8_W"]), params_pre["fc8_b"])
fc8 = tf.nn.relu(fc8)


# Run session
with tf.Session() as sess:
	# Initialise the variables and run
	init = tf.global_variables_initializer()
	sess.run(init)
	
	# with tf.device("/cpu:0"):
	with tf.device("/gpu:0"):
		fc8_eval = fc8.eval()[0]
		print(np.argmax(fc8_eval))

		conv23_eval = conv52.eval()[0]
		conv23_eval = np.swapaxes(conv23_eval, 0, 2) # (16(x),16(y),64)
		conv23_eval = np.swapaxes(conv23_eval, 1, 2) # (64,16(x),16(y))

		# for rec in conv23_eval[0:4]:
		# 	plt.imshow(rec, cmap="gray")
		# 	plt.show()
