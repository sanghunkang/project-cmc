#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages
import _pickle as cPickle

# Import external packages
import numpy as np
import tensorflow as tf

from arxtectInceptionv1 import arxtect_inceptionv1


FPATH_DATA_WEIGHTPRETRAINED = "../../../../dev-data/weightPretrained/googlenet.npy"
FPATH_DATA_TRAIN =  "../../../../dev-data/pickle/data_train.pickle"
FPATH_DATA_TEST =  "../../../../dev-data/pickle/data_test.pickle"

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
	return {X: batch[:,:len_input], y: batch[:,len_input:]}

# Inception-v1
print("++++++++++ Inception-v1 ++++++++++")
dict_lyr = np.load(FPATH_DATA_WEIGHTPRETRAINED, encoding='latin1').item() # return dict
params_pre = reformat_params(dict_lyr)

data_saved = {'var_epoch_saved': tf.Variable(0)}
params = {
	# len_ftmap_end = int(shape_ftmap_end[1]*shape_ftmap_end[2]*shape_ftmap_end[3])
	'fc6_W': tf.Variable(tf.random_normal([1024, 4096]), name='fc6_W'),
	'fc6_b': tf.Variable(tf.random_normal([4096]), name='fc6_b'),

	'fc7_W': tf.Variable(tf.random_normal([4096, 4096]), name='fc7_W'),
	'fc7_b': tf.Variable(tf.random_normal([4096]), name='fc7_b'),
	
	'fc8_W': tf.Variable(tf.random_normal([4096, 2]), name='fc8_W'),
	'fc8_b': tf.Variable(tf.random_normal([2]), name='fc8_b'), # 2 outputs (class prediction)
}

# tf Graph input
len_input = 224*224*3
num_class = 2 # Normal or Abnormal

X = tf.placeholder(tf.float32, [None, len_input])
y = tf.placeholder(tf.float32, [None, num_class])

pred = arxtect_inceptionv1(X, params_pre, params)

# BUILDING THE COMPUTATIONAL GRAPH
# Hyperparameters
learning_rate = 0.0001
num_itr = 100
batch_size = 512
display_step = 10

# Define loss and optimiser
crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
cost = tf.reduce_mean(crossEntropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
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
	
	# with tf.device("/cpu:0"):
	# with tf.device("/gpu:1"):
	# For train
	try:
		saver.restore(sess, './modelckpt/inception.ckpt')
		print('Model restored')
		epoch_saved = data_saved['var_epoch_saved'].eval()
	except tf.errors.NotFoundError:
		print('No saved model found')
		epoch_saved = 1
	except tf.errors.InvalidArgumentError:
		print('Model structure has change. Rebuild model')
		epoch_saved = 1

	# Training cycle
	for epoch in range(epoch_saved, epoch_saved + num_itr + 1):
		# Run optimization op (backprop)
		summary, acc_train, loss_train, _ = sess.run([merged, accuracy, cost, optimizer], feed_dict=feed_dict(data_train, batch_size, len_input))
		train_writer.add_summary(summary, epoch)

		summary, acc_test = sess.run([merged, accuracy], feed_dict=feed_dict(data_test, batch_size, len_input))
		test_writer.add_summary(summary, epoch)
		print("Accuracy at step {0}: {1}".format(epoch, acc_test))

		if epoch % display_step == 0:
			print("Epoch {0}, Minibatch Loss= {1:.6f}, Train Accuracy= {2:.5f}".format(epoch, loss_train, acc_train))

	print("Optimisation Finished!")

	# Save the variables
	epoch_new = epoch_saved + num_itr
	sess.run(data_saved["var_epoch_saved"].assign(epoch_saved + num_itr))
	fpath_ckpt = saver.save(sess, "./modelckpt/inception.ckpt")
	print("Model saved in file: {0}".format(fpath_ckpt))