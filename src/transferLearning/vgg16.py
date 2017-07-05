#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
img = img.resize((512,512))
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
# Network Parameters
len_input = 4096*3 # 64*64*3
n_classes = 2 # cat or dog

# tf Graph input
X = tf.reshape(arr_img, shape=[-1, 512, 512, 3])
# x = tf.placeholder(tf.float32, [None, len_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

data_saved = {'var_epoch_saved': tf.Variable(0)}


# Convolution and max pooling(down-sampling) Layers
# Parameters are from pretrained data
X_reshaped = tf.reshape(X, shape=[-1, 512, 512, 3])

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

# fc6 = tf.reshape(conv5p, [-1, params_pre["fc6_W"].get_shape().as_list()[0]])
# print(fc6.get_shape())

# fc6 = tf.add(tf.matmul(fc6, params_pre["fc6_W"]), params_pre["fc6_b"])
# fc6 = tf.nn.relu(fc6)
# # fc1 = tf.nn.dropout(fc1, params['dropout'])

# fc7 = tf.add(tf.matmul(fc6, params_pre["fc7_W"]), params_pre["fc7_b"])
# fc7 = tf.nn.relu(fc7)

# fc8 = tf.add(tf.matmul(fc7, params_pre["fc8_W"]), params_pre["fc8_b"])
# fc8 = tf.nn.relu(fc8)

# Fully connected layer (and Apply Dropout)
# Training is done here
shape_ftmap_end = conv5p.get_shape()
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

	# Dropout
	'dropout': 0.9 # Dropout, probability to keep units
}

fc6 = tf.reshape(conv5p, [-1, params["fc6_W"].get_shape().as_list()[0]])

fc6 = tf.add(tf.matmul(fc6, params["fc6_W"]), params["fc6_b"])
fc6 = tf.nn.relu(fc6)

fc7 = tf.add(tf.matmul(fc6, params["fc7_W"]), params["fc7_b"])
fc7 = tf.nn.relu(fc7)

fc8 = tf.add(tf.matmul(fc7, params["fc8_W"]), params["fc8_b"])
fc8 = tf.nn.relu(fc8)
pred = fc8

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

# Run session
with tf.Session() as sess:
	# Initialise the variables and run
	init = tf.global_variables_initializer()
	sess.run(init)
	
	# with tf.device("/cpu:0"):
	with tf.device("/gpu:0"):
		# For training
		try:
			saver.restore(sess, '.\\model\\model.ckpt')
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
		for epoch in range(epoch_saved, epoch_saved + training_epochs):
			batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]
			batch_x = batch[:, :len_input]
			batch_y = batch[:, len_input:]
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: params['dropout']})
			if epoch % display_step == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
		
				# batch_test = data_test[np.random.choice(data_test.shape[0], size=batch_size,  replace=True)]
				# Validation
				# acc_test = sess.run(accuracy, feed_dict={x: batch_test[:,:4096*3], y: batch_test[:,4096*3:], keep_prob: 1.})
				print('Epoch ' + str(epoch) + ', Minibatch Loss= ' + '{:.6f}'.format(loss) + ', Training Accuracy= ' + '{:.5f}'.format(acc))# + ', Validation Accuracy= ' + '{:.5f}'.format(acc_test))

				# batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]

			# if epoch % 10 == 0:  # Record summaries and test-set accuracy
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, epoch)
			print('Accuracy at step %s: %s' % (epoch, acc))
			# else:  # Record train set summaries, and train
			summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict(True))
			train_writer.add_summary(summary, epoch)
	
		print('Optimisation Finished!')

		# Save the variables
		epoch_new = epoch_saved + training_epochs
		sess.run(data_saved['var_epoch_saved'].assign(epoch_saved + training_epochs))
		print(data_saved['var_epoch_saved'].eval())
		save_path = saver.save(sess, '.\\model\\model.ckpt')
		print('Model saved in file: %s' % save_path)


		# For showing
		# fc8_eval = fc8.eval()[0]
		# print(np.argmax(fc8_eval))

		# conv23_eval = conv22.eval()[0]
		# conv23_eval = np.swapaxes(conv23_eval, 0, 2) # (16(x),16(y),64)
		# conv23_eval = np.swapaxes(conv23_eval, 1, 2) # (64,16(x),16(y))

		# for rec in conv23_eval[0:4]:
		# 	plt.imshow(rec, cmap="gray")
		# 	plt.show()
