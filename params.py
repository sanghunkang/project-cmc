#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import 3rd party packages
import tensorflow as tf

params= {
	# len_ftmap_end = int(shape_ftmap_end[1]*shape_ftmap_end[2]*shape_ftmap_end[3])
	'fc6_W': tf.Variable(tf.random_normal([1024, 4096]), name='fc6_W'),
	'fc6_b': tf.Variable(tf.random_normal([4096]), name='fc6_b'),

	'fc7_W': tf.Variable(tf.random_normal([4096, 4096]), name='fc7_W'),
	'fc7_b': tf.Variable(tf.random_normal([4096]), name='fc7_b'),

	'fc8_W': tf.Variable(tf.random_normal([4096, 4]), name='fc8_W'),
	'fc8_b': tf.Variable(tf.random_normal([4]), name='fc8_b'),  # 2 outputs (class prediction)
}
