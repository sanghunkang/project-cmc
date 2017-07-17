#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages
import _pickle as cPickle

# Import external packages
import numpy as np
import tensorflow as tf

from model_inceptionv1 import model_inceptionv1

DIR_DATA_PICKLE = "C:\\dev-data\\pickle\\"
DIR_DATA_WEIGHTPRETRAINED = "C:\\dev-data\\weightPretrained\\"

FPAHT_DATA_TRAIN =  "C:\\dev-data\\pickle\\data_train.pickle"
FPAHT_DATA_TEST =  "C:\\dev-data\\pickle\\data_test.pickle"
# DIR_DATA_WEIGHTPRETRAINED = "../../../../dev-data/weightPretrained/"

# Define some functions... for whatever purposes
def read_data(fpath):
	"""
	args:
		fpath: str or pathlike object
	return:
		data: nparray
	"""
	with open(fpath, "rb") as fo:
		data_train = cPickle.load(fo, encoding="bytes")
		np.random.shuffle(data_train)
	return data_train

def reformat_params(dict_lyr):
	params_pre = {}
	for key in dict_lyr:
		params_pre[key + "_W"] = tf.Variable(dict_lyr[key]["weights"], name=key + "_W")
		params_pre[key + "_b"] = tf.Variable(dict_lyr[key]["biases"], name=key + "_b")
	return params_pre

def feed_dict(data, batch_size):
	batch = data[np.random.choice(data.shape[0], size=batch_size,  replace=True)]
	return {X: batch[:,:len_input], y: batch[:,len_input:]}

# Inception-v1
print("++++++++++ Inception-v1 ++++++++++")
dict_lyr = np.load(DIR_DATA_WEIGHTPRETRAINED + "googlenet.npy", encoding = 'latin1').item() # return dict
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
n_classes = 2 # Normal or Abnormal

X = tf.placeholder(tf.float32, [None, len_input])
y = tf.placeholder(tf.float32, [None, n_classes])

pred = model_inceptionv1(X, params_pre, params)

# BUILDING THE COMPUTATIONAL GRAPH
# Hyperparameters
learning_rate = 0.0001
n_itr = 1000
batch_size = 128
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
	data_train = read_data(DIR_DATA_PICKLE + "data_train.pickle")
	data_test = read_data(DIR_DATA_PICKLE + "data_test.pickle")

	summaries_dir = '.\\logs'
	train_writer = tf.summary.FileWriter(summaries_dir + '\\train', sess.graph)
	test_writer = tf.summary.FileWriter(summaries_dir + '\\test')
	
	# Initialise the variables and run
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
			epoch_saved = 1
		except tf.errors.InvalidArgumentError:
			print('Model structure has change. Rebuild model')
			epoch_saved = 1

		# Training cycle
		for epoch in range(epoch_saved, epoch_saved + n_itr + 1):
			# Run optimization op (backprop)
			summary, acc_train, loss_train, _ = sess.run([merged, accuracy, cost, optimizer], feed_dict=feed_dict(data_train, batch_size))
			train_writer.add_summary(summary, epoch)
			
			summary, acc_test = sess.run([merged, accuracy], feed_dict=feed_dict(data_test, batch_size))
			test_writer.add_summary(summary, epoch)
			print("Accuracy at step {0}: {1}".format(epoch, acc_test))

			if epoch % display_step == 0:
				print("Epoch {0}, Minibatch Loss= {1:.6f}, Train Accuracy= {2:.5f}".format(epoch, loss_train, acc_train))
	
		print("Optimisation Finished!")

		# Save the variables
		epoch_new = epoch_saved + n_itr
		sess.run(data_saved["var_epoch_saved"].assign(epoch_saved + n_itr))
		fpath_ckpt = saver.save(sess, ".\\modelckpt\\inception.ckpt")
		print("Model saved in file: {0}".format(fpath_ckpt))