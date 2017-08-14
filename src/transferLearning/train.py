#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages
import _pickle as cPickle
import time

# Import external packages
import numpy as np
import tensorflow as tf

from params import params
from arxtectInceptionv1 import arxtect_inceptionv1


FPATH_DATA_WEIGHTPRETRAINED = "../../../../dev-data/weightPretrained/googlenet.npy"
FPATH_DATA_TRAIN =  "../../../../dev-data/pickle/data_train_th20.pickle"
FPATH_DATA_TEST =  "../../../../dev-data/pickle/data_test_th20.pickle"

# Define some functions... for whatever purposes
def read_data(fpath):
	"""
	args:
		fpath 		: str or pathlike object
	return:
		data 		: np.array
	"""
	with open(fpath, "rb") as fo:
		data_train = cPickle.load(fo, encoding="bytes")
		np.random.shuffle(data_train)
	return data_train

def reformat_params(dict_lyr):
	"""
	Convert {layer:{Weight, bias}} into {layer_Weight, layer_bias} for easier referencing

	args:
		dict_lyr 	: dict, {layer_name:{variable1_name: tf.Variable, variable2_name: tf.Variable}}
	return:
		params_pre 	: dict, {variable_name: tf.Variable}
	"""
	params_pre = {}
	for key in dict_lyr:
		params_pre[key + "_W"] = tf.Variable(dict_lyr[key]["weights"], name=key + "_W")
		params_pre[key + "_b"] = tf.Variable(dict_lyr[key]["biases"], name=key + "_b")
	return params_pre

def feed_dict(data, batch_size, len_input):
	"""
	args:
		data 		: np.array, 2-dimensional
		batch_size 	: int
		len_input 	: int
	return:
					: dict, {X: np.array of shape(len_input, batch_size), y: np.array of shape(num_class, batch_size)}
	"""
	batch = data[np.random.choice(data.shape[0], size=batch_size,  replace=True)]
	return {X0: batch[:batch_size//2, :len_input],
			X1: batch[batch_size//2:, :len_input],
			y0: batch[:batch_size//2, len_input:],
			y1: batch[batch_size//2:, len_input:]}

# def merge_gradients(grad0, grad1):
# 	grad_merged = {}
# 	for i, pair in enumerate(grad0): grad0[i] = tf.add grad0[i][0] + grad1[i][0]
# 		grad_merged[a] = 0.5


# Inception-v1
print("++++++++++ Inception-v1 ++++++++++")
dict_lyr = np.load(FPATH_DATA_WEIGHTPRETRAINED, encoding='latin1').item() # return dict
params_pre = reformat_params(dict_lyr)

data_saved = {'var_epoch_saved': tf.Variable(0)}
# params = {
# 	# len_ftmap_end = int(shape_ftmap_end[1]*shape_ftmap_end[2]*shape_ftmap_end[3])
# 	'fc6_W': tf.Variable(tf.random_normal([4096, 4096]), name='fc6_W'),
# 	'fc6_b': tf.Variable(tf.random_normal([4096]), name='fc6_b'),
#
# 	'fc7_W': tf.Variable(tf.random_normal([4096, 4096]), name='fc7_W'),
# 	'fc7_b': tf.Variable(tf.random_normal([4096]), name='fc7_b'),
#
# 	'fc8_W': tf.Variable(tf.random_normal([4096, 2]), name='fc8_W'),
# 	'fc8_b': tf.Variable(tf.random_normal([2]), name='fc8_b'), # 2 outputs (class prediction)
# }

# BUILDING THE COMPUTATIONAL GRAPH
# Hyperparameters
learning_rate = 0.0001
num_itr = 500
batch_size = 128
display_step = 10

# tf Graph input
len_input = 448*448*3
num_class = 2 # Normal or Abnormal

X0 = tf.placeholder(tf.float32, [None, len_input])
y0 = tf.placeholder(tf.float32, [None, num_class])
X1 = tf.placeholder(tf.float32, [None, len_input])
y1 = tf.placeholder(tf.float32, [None, num_class])

with tf.device("/gpu:0"):
	# Define loss, compute gradients
	pred0 = arxtect_inceptionv1(X0, params_pre, params)
	crossEntropy0 = tf.nn.softmax_cross_entropy_with_logits(logits=pred0, labels=y0)
	cost0 = tf.reduce_mean(crossEntropy0)
	grad0 = tf.train.AdamOptimizer(learning_rate=learning_rate).compute_gradients(cost0)

	# Evaluate
	correct_pred0 = tf.equal(tf.argmax(pred0, 1), tf.argmax(y0, 1))

with tf.device("/gpu:1"):
	# Define loss, compute gradients
	pred1 = arxtect_inceptionv1(X1, params_pre, params)
	crossEntropy1 = tf.nn.softmax_cross_entropy_with_logits(logits=pred1, labels=y1)
	cost1 = tf.reduce_mean(crossEntropy1)
	grad1 = tf.train.AdamOptimizer(learning_rate=learning_rate).compute_gradients(cost1)

	# Evaluate
	correct_pred1 = tf.equal(tf.argmax(pred1, 1), tf.argmax(y1, 1))

	# Merging computed gradients and cost (hopefully)
	grad = grad0 + grad1
	optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(grad)

	cost = 0.5*(cost0 + cost1)

# Evaluate model
correct_pred = tf.concat([correct_pred0, correct_pred1], axis=0)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Integrate tf summaries
tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

# RUNNING THE COMPUTATIONAL GRAPH
# Define saver
saver = tf.train.Saver()

# Configure memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Run session
with tf.Session(config=config) as sess:
	data_train = read_data(FPATH_DATA_TRAIN)
	data_test = read_data(FPATH_DATA_TEST)

	summaries_dir = './logs'
	train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(summaries_dir + '/test')

	# Initialise the variables and run
	init = tf.global_variables_initializer()
	sess.run(init)

	# For train
	try:
		saver.restore(sess, './modelckpt/inception500.ckpt')
		print('Model restored')
		epoch_saved = data_saved['var_epoch_saved'].eval()
	except tf.errors.NotFoundError:
		print('No saved model found')
		epoch_saved = 0
	except tf.errors.InvalidArgumentError:
		print('Model structure has change. Rebuild model')
		epoch_saved = 0

	# Training cycle
	t0 = time.time()
	for epoch in range(epoch_saved, epoch_saved + num_itr):
		# Run optimization op (backprop)
		summary, acc_train, loss_train, _ = sess.run([merged, accuracy, cost, optimizer1], feed_dict=feed_dict(data_train, batch_size, len_input))
		train_writer.add_summary(summary, epoch)

		summary, acc_test = sess.run([merged, accuracy], feed_dict=feed_dict(data_test, batch_size, len_input))
		test_writer.add_summary(summary, epoch)
		print("Accuracy at step {0}: {1}".format(epoch, acc_test))

		if epoch % display_step == 0:
			print("Epoch {0}, Minibatch Loss= {1:.6f}, Train Accuracy= {2:.5f}".format(epoch, loss_train, acc_train))

	print("Optimisation Finished!")
	t1 = time.time()
	print(t1-t0)

	# Save the variables
	epoch_new = epoch_saved + num_itr
	sess.run(data_saved["var_epoch_saved"].assign(epoch_new))
	fpath_ckpt = saver.save(sess, "./modelckpt/inception{0}.ckpt".format(epoch_new))
	print("Model saved in file: {0}".format(fpath_ckpt))