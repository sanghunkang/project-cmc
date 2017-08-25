#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages

# Import external packages
import tensorflow as tf

def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def arxt_vgg16(X, params_pre, params):	
	X_reshaped = tf.reshape(X, shape=[-1, 224, 224, 3])

	conv11 = conv2d(X_reshaped, params_pre['conv1_1_W'], params_pre['conv1_1_b'])
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

	fc6 = tf.reshape(conv5p, [-1, params["fc6_W"].get_shape().as_list()[0]])

	fc6 = tf.add(tf.matmul(fc6, params["fc6_W"]), params["fc6_b"])
	fc6 = tf.nn.relu(fc6)

	fc7 = tf.add(tf.matmul(fc6, params["fc7_W"]), params["fc7_b"])
	fc7 = tf.nn.relu(fc7)

	fc8 = tf.add(tf.matmul(fc7, params["fc8_W"]), params["fc8_b"])
	pred = tf.nn.relu(fc8)
	return pred