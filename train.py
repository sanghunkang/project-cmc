#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages
import pickle as cPickle
import time

from functools import reduce
# Import external packages
import numpy as np
import tensorflow as tf

from params import params
from arxtectInceptionv1 import arxtect_inceptionv1

# Define some functions... for whatever purposes
def read_data(fpath):
	"""
	args:
		fpath 		: str or pathlike object
	return:
		data 		: np.array
	"""
	with open(fpath, "rb") as fo:
		data = cPickle.load(fo)
		np.random.shuffle(data)
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

def feed_dict(stack_data, batch_size, len_input):
	"""
	args:
		data 		: np.array, 2-dimensional
		batch_size 	: int
		len_input 	: int
	return:
					: dict, {X: np.array of shape(len_input, batch_size), y: np.array of shape(num_class, batch_size)}
	"""
	assert batch_size%len(stack_data)==0
	batch_size_each = batch_size//len(stack_data)
	batch = np.zeros(shape=(batch_size, len_input+len(stack_data)), dtype=np.float32)
	for i, data in enumerate(stack_data):
		batch[i*batch_size_each:(i + 1)*batch_size_each] = data[np.random.choice(data.shape[0], size=batch_size_each,  replace=True)]
	# batch = data[np.random.choice(data.shape[0], size=batch_size,  replace=True)]
	return {X: batch[:,:len_input], y: batch[:,len_input:]}

# Inception-v1
FPATH_DATA_WEIGHTPRETRAINED = "../../dev-data/weight-pretrained/googlenet.npy"
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("dir_data_train", "../../dev-data/project-cmc/pickle/train.", "The directory to save the model files in.")
tf.flags.DEFINE_string("dir_data_validation", "../../dev-data/project-cmc/pickle/validation", "The directory to save the model files in.")
tf.flags.DEFINE_string("ckpt_name","ckpt","name of ckpt file")
tf.flags.DEFINE_integer("num_class", 2, "Learning rate, epsilon")
tf.flags.DEFINE_integer("batch_size", 128, "How many examples to process per batch for training and evaluation")
tf.flags.DEFINE_integer("num_steps", 1000, "How many times to update weights")
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate, epsilon")

tf.flags.DEFINE_integer("first_gpu_id", 0, "id of the first gpu")
tf.flags.DEFINE_integer("num_gpu",1, "Number of gpu to utilise")
print("++++++++++ Inception-v1 ++++++++++")
dict_lyr = np.load(FPATH_DATA_WEIGHTPRETRAINED, encoding='latin1').item() # return dict
params_pre = reformat_params(dict_lyr)

data_saved = {'var_epoch_saved': tf.Variable(0)}


# BUILDING THE COMPUTATIONAL GRAPH
# Hyperparameters
learning_rate = FLAGS.learning_rate
display_step = 10

# tf Graph input
len_input = 224*224*3
num_class = FLAGS.num_class # Normal or Abnormal
num_device = FLAGS.num_gpu


with tf.device("/gpu:{0}".format(FLAGS.first_gpu_id)):
	X = tf.placeholder(tf.float32, [None, len_input])
	y = tf.placeholder(tf.float32, [None, num_class])

	stack_X = tf.split(X, num_device, 0)
	stack_y = tf.split(y, num_device, 0)
	stack_pred=[0]*num_device
	stack_xentropy=[0]*num_device
	stack_cost=[0]*num_device
	stack_grad=[0]*num_device
for i in range(num_device):

	with tf.device("/gpu:{0}".format(i + FLAGS.first_gpu_id)):
		# Define loss, compute gradients
		stack_pred[i] = arxtect_inceptionv1(stack_X[i], params_pre, params)
		stack_xentropy[i] = tf.nn.softmax_cross_entropy_with_logits(logits=stack_pred[i], labels=stack_y[i])
		stack_cost[i] = tf.reduce_mean(stack_xentropy[i])
		stack_grad[i] = tf.train.AdamOptimizer(learning_rate=learning_rate).compute_gradients(stack_cost[i])
	
with tf.device("/gpu:{0}".format(i + FLAGS.first_gpu_id)):
	#print(stack_grad[0])
	grad = reduce(lambda x0, x1: x0 + x1, stack_grad) 
	#grad = stack_grad[0] + stack_grad[1]
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(grad)

	# Evaluate model
	pred = tf.concat(stack_pred, axis=0)
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	cost = tf.reduce_mean(stack_xentropy)

# Integrate tf summaries
tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

# RUNNING THE COMPUTATIONAL GRAPH
def main(unused_argv):

	# Define saver
	saver = tf.train.Saver()

	# Configure memory growth
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	# Run session
	with tf.Session(config=config) as sess:
		num_itr = FLAGS.num_steps
		batch_size = FLAGS.batch_size

		data_train = read_data(FLAGS.fpath_data_train)
		data_test = read_data(FLAGS.fpath_data_validation)
		print(data_train.shape)
		print(data_test.shape)

		summaries_dir = './{0}'.format(FLAGS.ckpt_name)
		train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(summaries_dir + '/test')

		# Initialise the variables and run
		init = tf.global_variables_initializer()
		sess.run(init)

		# For train
		try:
			saver.restore(sess, './{}/checkpoint.ckpt'.format(FLAGS.ckpt_name))
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
			summary, acc_train, loss_train, _ = sess.run([merged, accuracy, cost, optimizer], feed_dict=feed_dict(data_train, batch_size, len_input))
			train_writer.add_summary(summary, epoch)

			summary, acc_test = sess.run([merged, accuracy], feed_dict=feed_dict(data_test, data_test.shape[0], len_input))
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
		fpath_ckpt = saver.save(sess, "./{0}/checkpoint.ckpt".format(FLAGS.ckpt_name))
		print("Model saved in file: {0}".format(fpath_ckpt))

if __name__ == "__main__":
	tf.app.run()
