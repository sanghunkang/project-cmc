#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# sess = tf.Session()
# imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

DIR_DATA_WEIGHTPRETRAINED = "C:\\dev-data\\weightPretrained\\"
DIR_DATA_WEIGHTPRETRAINED = "../../../../dev-data/weightPretrained/"

# VGGNet
print("++++++++++ VGGNet ++++++++++")
# aa = np.load("pretrained\\vgg16.npz")

def reformat_params(dict_lyr):
	dict_params = {}
	for key in dict_lyr:
		dict_params[key + "_W"] = tf.Variable(dict_lyr[key][0], name=key + "_W")
		dict_params[key + "_b"] = tf.Variable(dict_lyr[key][1], name=key + "_b")
	return dict_params

dict_lyr = np.load(DIR_DATA_WEIGHTPRETRAINED + "vgg16.npy", encoding = "latin1").item() # return dict
dict_params = reformat_params(dict_lyr)

keys = []
for x in dict_params: keys.append(x)

keys.sort()


# GoogLeNet
# print("++++++++++ GoogLeNet ++++++++++")
# aa = np.load("pretrained\\googlenet.npy", encoding = 'latin1').item() # return dict
# for a in aa: print(a, aa[a]["biases"].shape, aa[a]["weights"].shape)

import PIL.Image as Image

img = Image.open("../../data_light/bmp/I0000001.BMP")
img = img.resize((224,224))
arr_img = np.asarray(img, dtype=np.float32)
print(arr_img.shape)
print(arr_img.dtype)

for key in keys: print(key, dict_params[key])


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

# Convolution and max pooling(down-sampling) Layer
conv11 = conv2d(tsr_X, dict_params['conv1_1_W'], dict_params['conv1_1_b'])
conv12 = conv2d(conv11, dict_params['conv1_2_W'], dict_params['conv1_2_b'])
# conv13 = conv2d(conv12, dict_params['conv1_3_W'], dict_params['conv1_3_b'])


# conv13 = maxpool2d(conv13, k=2)

# # Convolution and max pooling(down-sampling) Layer
# conv21 = conv2d(conv13, dict_params['conv1_1_W'], dict_params['b_conv21'])
# conv22 = conv2d(conv21, dict_params['W_conv22'], dict_params['b_conv22'])
# conv23 = conv2d(conv22, dict_params['W_conv23'], dict_params['b_conv23'])
# conv23 = maxpool2d(conv23, k=2)

with tf.Session() as sess:
	# Initialise the variables and run
	init = tf.global_variables_initializer()
	sess.run(init)
	
	with tf.device("/cpu:0"):
		conv23 = conv12.eval()[0]
		conv23 = np.swapaxes(conv23, 0, 2) # (16(x),16(y),64)
		conv23 = np.swapaxes(conv23, 1, 2) # (64,16(x),16(y))

		for rec in conv23[0:4]:
			plt.imshow(rec)
			plt.show()
