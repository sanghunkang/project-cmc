#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

# sess = tf.Session()
# imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

DIR_DATA_WEIGHTPRETRAINED = "C:\\dev-data\\weightPretrained\\"

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
for key in keys: print(key, dict_params[key])

# GoogLeNet
# print("++++++++++ GoogLeNet ++++++++++")
# aa = np.load("pretrained\\googlenet.npy", encoding = 'latin1').item() # return dict
# for a in aa: print(a, aa[a]["biases"].shape, aa[a]["weights"].shape)


##############################################################################
# def feed_dict(train):
# 	"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
# 	if train:
# 		xs, ys = batch_x, batch_y
# 		k = params['dropout']
# 	else:
# 		batch_test = data_test[np.random.choice(data_test.shape[0], size=batch_size,  replace=True)]
# 		xs, ys =  batch_test[:,:4096*3], batch_test[:,4096*3:]
# 		k = 1.0
# 	return {x: xs, y: ys, keep_prob: k}

# # RUNNING THE COMPUTATIONAL GRAPH
# # Define saver 
# merged = tf.summary.merge_all()
# saver = tf.train.Saver()

# # Launch the graph
# with tf.Session() as sess:
# 	summaries_dir = '.\\logs'
# 	train_writer = tf.summary.FileWriter(summaries_dir + '\\train', sess.graph)
# 	test_writer = tf.summary.FileWriter(summaries_dir + '\\test')
# 	tf.global_variables_initializer().run()
# 	step = 1

# 	with tf.device('/gpu:0'):
# 		# Restore saved model if any
# 		try:
# 			saver.restore(sess, '.\\model\\model.ckpt')
# 			print('Model restored')
# 			epoch_saved = data_saved['var_epoch_saved'].eval()
# 		except tf.errors.NotFoundError:
# 			print('No saved model found')
# 			epoch_saved = 0
# 		except tf.errors.InvalidArgumentError:
# 			print('Model structure has change. Rebuild model')
# 			epoch_saved = 0

# 		# Training cycle
# 		print(epoch_saved)
# 		# batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]
# 		for epoch in range(epoch_saved, epoch_saved + training_epochs):
# 			batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]
# 			batch_x = batch[:, :4096*3]
# 			batch_y = batch[:, 4096*3:]
# 			# Run optimization op (backprop)
# 			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: params['dropout']})
# 			if epoch % display_step == 0:
# 				# Calculate batch loss and accuracy
# 				loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
		
# 				# batch_test = data_test[np.random.choice(data_test.shape[0], size=batch_size,  replace=True)]
# 				# Validation
# 				# acc_test = sess.run(accuracy, feed_dict={x: batch_test[:,:4096*3], y: batch_test[:,4096*3:], keep_prob: 1.})
# 				print('Epoch ' + str(epoch) + ', Minibatch Loss= ' + '{:.6f}'.format(loss) + ', Training Accuracy= ' + '{:.5f}'.format(acc))# + ', Validation Accuracy= ' + '{:.5f}'.format(acc_test))

# 				# batch = data_training[np.random.choice(data_training.shape[0], size=batch_size,  replace=True)]

# 			# if epoch % 10 == 0:  # Record summaries and test-set accuracy
# 			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
# 			test_writer.add_summary(summary, epoch)
# 			print('Accuracy at step %s: %s' % (epoch, acc))
# 			# else:  # Record train set summaries, and train
# 			summary, _ = sess.run([merged, optimizer], feed_dict=feed_dict(True))
# 			train_writer.add_summary(summary, epoch)
	
# 		print('Optimisation Finished!')

# 		# Save the variables
# 		epoch_new = epoch_saved + training_epochs
# 		sess.run(data_saved['var_epoch_saved'].assign(epoch_saved + training_epochs))
# 		print(data_saved['var_epoch_saved'].eval())
# 		save_path = saver.save(sess, '.\\model\\model.ckpt')
# 		print('Model saved in file: %s' % save_path)
