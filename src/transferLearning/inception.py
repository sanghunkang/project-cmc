#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _pickle as cPickle

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import tensorflow as tf

# img = Image.open("../../data_light/bmp/I0000001.BMP")
# img = Image.open("..\\..\\data_light/bmp/I0000001.BMP")
# img = img.resize((224, 224))
# arr_img = np.asarray(img, dtype=np.float32)
# print(arr_img.shape)
# print(arr_img.dtype)

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

# Network Parameters
len_input = 224*224*3 # 64*64*3
n_classes = 2 # cat or dog

# tf Graph input
# X = tf.reshape(arr_img, shape=[-1, 224, 224, 3])
X = tf.placeholder(tf.float32, [None, len_input])
y = tf.placeholder(tf.float32, [None, n_classes])
# keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

data_saved = {'var_epoch_saved': tf.Variable(0)}


# Model
X_reshaped = tf.reshape(X, shape=[-1, 224, 224, 3])
conv1_7x7_s2 = conv2d(X_reshaped, params_pre['conv1_7x7_s2_W'], params_pre['conv1_7x7_s2_b'], strides=2)
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

# inception_fc = tf.reshape(inception_5ap, [-1, params_pre["loss3_classifier_W"].get_shape().as_list()[0]])
# print(inception_fc.get_shape())
# linclassifier = tf.add(tf.matmul(inception_fc, params_pre["loss3_classifier_W"]), params_pre["loss3_classifier_b"])
# linclassifier = tf.nn.relu(linclassifier)
# fc1 = tf.nn.dropout(fc1, params['dropout'])

# Fully connected layer (and Apply Dropout)
# Train is done here
shape_ftmap_end = inception_5ap.get_shape()
len_ftmap_end = int(shape_ftmap_end[1]*shape_ftmap_end[2]*shape_ftmap_end[3])

params = {
	# Variables	
	'fc6_W': tf.Variable(tf.random_normal([len_ftmap_end, 4096]), name='fc6_W'),
	'fc6_b': tf.Variable(tf.random_normal([4096]), name='fc6_b'),

	'fc7_W': tf.Variable(tf.random_normal([4096, 4096]), name='fc7_W'),
	'fc7_b': tf.Variable(tf.random_normal([4096]), name='fc7_b'),
	
	# 2 outputs (class prediction)
	'fc8_W': tf.Variable(tf.random_normal([4096, 2]), name='fc8_W'),
	'fc8_b': tf.Variable(tf.random_normal([2]), name='fc8_b'),

	# # Dropout
	# 'dropout': 0.9 # Dropout, probability to keep units
}

fc6 = tf.reshape(inception_5ap, [-1, params["fc6_W"].get_shape().as_list()[0]])

fc6 = tf.add(tf.matmul(fc6, params["fc6_W"]), params["fc6_b"])
fc6 = tf.contrib.layers.batch_norm(fc6)
fc6 = tf.nn.relu(fc6)

fc7 = tf.add(tf.matmul(fc6, params["fc7_W"]), params["fc7_b"])
fc7 = tf.contrib.layers.batch_norm(fc7)
fc7 = tf.nn.relu(fc7)

fc8 = tf.add(tf.matmul(fc7, params["fc8_W"]), params["fc8_b"])
fc8 = tf.contrib.layers.batch_norm(fc8)
fc8 = tf.nn.relu(fc8)
pred = fc8

# BUILDING THE COMPUTATIONAL GRAPH
# Parameters
learning_rate = 0.0001
train_epochs = 1200
batch_size = 128
display_step = 10


with open("C:\\dev-data\\pickle\\data_train.pickle", "rb") as fo:
	data_train = cPickle.load(fo, encoding="bytes")

with open("C:\\dev-data\\pickle\\data_test.pickle", "rb") as fo:
	data_test = cPickle.load(fo, encoding="bytes")
np.random.shuffle(data_train)
np.random.shuffle(data_test)

def feed_dict(src_feed):
	if src_feed == "train":
		xs, ys = batch_x, batch_y
	elif src_feed == "test":
		batch_test = data_test[np.random.choice(data_test.shape[0], size=batch_size,  replace=True)]
		xs, ys =  batch_test[:,:len_input], batch_test[:,len_input:]
	return {X: xs, y: ys}

# Define loss and optimiser
crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(crossEntropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# RUNNING THE COMPUTATIONAL GRAPH
# Define saver 
merged = tf.summary.merge_all()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Run session
with tf.Session(config=config) as sess:
	# Initialise the variables and run
	summaries_dir = '.\\logs'
	train_writer = tf.summary.FileWriter(summaries_dir + '\\train', sess.graph)
	test_writer = tf.summary.FileWriter(summaries_dir + '\\test')
	init = tf.global_variables_initializer()
	sess.run(init)
	
	# with tf.device("/cpu:0"):
	with tf.device("/gpu:0"):
		# For train
		try:
			saver.restore(sess, '.\\modelckpt\\inception.ckpt')
			print('Model restored')
			epoch_saved = data_saved['var_epoch_saved'].eval()
		except tf.errors.NotFoundError:
			print('No saved model found')
			epoch_saved = 0
		except tf.errors.InvalidArgumentError:
			print('Model structure has change. Rebuild model')
			epoch_saved = 0

		# Training cycle
		print(epoch_saved)
		for epoch in range(epoch_saved, epoch_saved + train_epochs):
			batch = data_train[np.random.choice(data_train.shape[0], size=batch_size,  replace=True)]
			batch_x = batch[:, :len_input]
			batch_y = batch[:, len_input:]
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
			if epoch % display_step == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x, y: batch_y})
		
				print('Epoch ' + str(epoch) + ', Minibatch Loss= ' + '{:.6f}'.format(loss) + ', Train Accuracy= ' + '{:.5f}'.format(acc))# + ', Validation Accuracy= ' + '{:.5f}'.format(acc_test))


			# if epoch % 10 == 0:  # Record summaries and test-set accuracy
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict("test"))
			test_writer.add_summary(summary, epoch)
			print('Accuracy at step %s: %s' % (epoch, acc))
			# else:  # Record train set summaries, and train
			summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict("train"))
			train_writer.add_summary(summary, epoch)
	
		print('Optimisation Finished!')

		# Save the variables
		epoch_new = epoch_saved + train_epochs
		sess.run(data_saved['var_epoch_saved'].assign(epoch_saved + train_epochs))
		print(data_saved['var_epoch_saved'].eval())
		save_path = saver.save(sess, '.\\modelckpt\\inception.ckpt')
		print('Model saved in file: %s' % save_path)