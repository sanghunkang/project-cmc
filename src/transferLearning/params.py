#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import 3rd party packages
import tensorflow as tf

params= {
	# Variables
	# 5x5 conv, 1 input, 32 outputs
	'W_conv11': tf.Variable(tf.random_normal([7, 7, 3, 32]), name='W_conv11'),
	'W_conv12': tf.Variable(tf.random_normal([7, 7, 32, 32]), name='W_conv12'),
	'W_conv13': tf.Variable(tf.random_normal([7, 7, 32, 32]), name='W_conv13'),
	
	'W_conv21': tf.Variable(tf.random_normal([3, 3, 32, 64]), name='W_conv21'),
	'W_conv22': tf.Variable(tf.random_normal([3, 3, 64, 64]), name='W_conv22'),
	'W_conv23': tf.Variable(tf.random_normal([3, 3, 64, 64]), name='W_conv23'),

	'W_conv31': tf.Variable(tf.random_normal([3, 3, 64, 128]), name='W_conv31'),
	'W_conv32': tf.Variable(tf.random_normal([3, 3, 128, 128]), name='W_conv32'),
	'W_conv33': tf.Variable(tf.random_normal([3, 3, 128, 128]), name='W_conv33'),

	# 'W_conv41': tf.Variable(tf.random_normal([3, 3, 128, 256]), name='W_conv41'),
	# 'W_conv42': tf.Variable(tf.random_normal([3, 3, 256, 256]), name='W_conv42'),
	# 'W_conv43': tf.Variable(tf.random_normal([3, 3, 256, 256]), name='W_conv43'),
	
	'W_fc1': tf.Variable(tf.random_normal([8*8*128, 4096]), name='W_fc1'),
	'W_fc2': tf.Variable(tf.random_normal([4096, 4096]), name='W_fc2'),
	
	# 4096 inputs, 2 outputs (class prediction)
	'W_out': tf.Variable(tf.random_normal([4096, 2]), name='W_out'),

	# Biases
	'b_conv11': tf.Variable(tf.random_normal([32]), name='b_conv11'),
	'b_conv12': tf.Variable(tf.random_normal([32]), name='b_conv12'),
	'b_conv13': tf.Variable(tf.random_normal([32]), name='b_conv13'),

	'b_conv21': tf.Variable(tf.random_normal([64]), name='b_conv21'),
	'b_conv22': tf.Variable(tf.random_normal([64]), name='b_conv22'),
	'b_conv23': tf.Variable(tf.random_normal([64]), name='b_conv23'),

	'b_conv31': tf.Variable(tf.random_normal([128]), name='b_conv31'),
	'b_conv32': tf.Variable(tf.random_normal([128]), name='b_conv32'),
	'b_conv33': tf.Variable(tf.random_normal([128]), name='b_conv33'),

	# 'b_conv41': tf.Variable(tf.random_normal([256]), name='b_conv41'),
	# 'b_conv42': tf.Variable(tf.random_normal([256]), name='b_conv42'),
	# 'b_conv43': tf.Variable(tf.random_normal([256]), name='b_conv43'),

	'b_fc1': tf.Variable(tf.random_normal([4096]), name='b_fc1'),
	'b_fc2': tf.Variable(tf.random_normal([4096]), name='b_fc2'),
	
	'b_out': tf.Variable(tf.random_normal([2]), name='b_out'),

	# Dropout
	'dropout': 0.9 # Dropout, probability to keep units
}