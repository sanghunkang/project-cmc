#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages
import os, pickle, time

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
        data = pickle.load(fo)
        np.random.shuffle(data)
    return data


def generate_stack_data(dirpath):
    print("____________________________________")
    for fpath in os.listdir(dirpath): print(fpath)
    print("____________________________________")
    return [read_data(os.path.join(dirpath, fpath)) for fpath in os.listdir(dirpath)]


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
    # print("++++++++++++++++++++++++++++++++++++++++")
    # print(len(stack_data))
    # print("++++++++++++++++++++++++++++++++++++++++")
    assert batch_size % len(stack_data) == 0, "Batch size must be a multiplication of the number of classes"

    batch_size_each = batch_size // len(stack_data)
    batch = np.zeros(shape=(batch_size, len_input + len(stack_data)), dtype=np.float32)
    for i, data in enumerate(stack_data):
        # print(i, batch_size_each, data.shape[0])
        batch[i * batch_size_each:(i + 1) * batch_size_each] = data[
            np.random.choice(data.shape[0], size=batch_size_each, replace=True)]
    np.random.shuffle(batch)
    # batch = data[np.random.choice(data.shape[0], size=batch_size,  replace=True)]
    return {X: batch[:, :len_input], y: batch[:, len_input:]}


# Inception-v1
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("dir_data_data", "../../dev-data/project-cmc/pickle/eval",
                       "Directory containing evaluation pickle files")
tf.flags.DEFINE_string("ckpt_name", "ckpt", "Name of ckpt file")
tf.flags.DEFINE_integer("batch_size", 128, "How many examples to process per batch for training and evaluation")
tf.flags.DEFINE_integer("num_steps", 1000, "How many times to update weights")
tf.flags.DEFINE_integer("display_step", 10, "(Should have been) How often to show logs")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate, epsilon")

tf.flags.DEFINE_integer("first_gpu_id", 0, "id of the first gpu")
tf.flags.DEFINE_integer("num_gpu", 1, "Number of gpu to utilise, even numbers are recommended")

# Read pretrained weights
dict_lyr = np.load("../../dev-data/weight-pretrained/googlenet.npy", encoding='latin1').item()  # return dict
params_pre = reformat_params(dict_lyr)

data_saved = {'var_epoch_saved': tf.Variable(0)}

# BUILDING THE COMPUTATIONAL GRAPH

# tf Graph input
len_input = 448 * 448 * 3
num_class = len(os.listdir(FLAGS.dir_data_train))  # Normal or Abnormal

with tf.device("/gpu:{0}".format(FLAGS.first_gpu_id)):
    X = tf.placeholder(tf.float32, [None, len_input])
    y = tf.placeholder(tf.float32, [None, num_class])

    stack_X = tf.split(X, FLAGS.num_gpu, 0)
    stack_y = tf.split(y, FLAGS.num_gpu, 0)
    stack_pred = [0] * FLAGS.num_gpu
    stack_xentropy = [0] * FLAGS.num_gpu
    stack_cost = [0] * FLAGS.num_gpu
    stack_grad = [0] * FLAGS.num_gpu
for i in range(FLAGS.num_gpu):
    with tf.device("/gpu:{0}".format(i + FLAGS.first_gpu_id)):
        # Define loss, compute gradients
        stack_pred[i] = arxtect_inceptionv1(stack_X[i], params_pre, params)
        stack_xentropy[i] = tf.nn.softmax_cross_entropy_with_logits(logits=stack_pred[i], labels=stack_y[i])
        stack_cost[i] = tf.reduce_mean(stack_xentropy[i])
        stack_grad[i] = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).compute_gradients(stack_cost[i])

with tf.device("/gpu:{0}".format(i + FLAGS.first_gpu_id)):
    # print(stack_grad[0])
    grad = reduce(lambda x0, x1: x0 + x1, stack_grad)
    # grad = stack_grad[0] + stack_grad[1]
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).apply_gradients(grad)

    # Evaluate model
    pred = tf.concat(stack_pred, axis=0)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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

        stack_data_train = generate_stack_data(FLAGS.dir_data_train)
        stack_data_eval = generate_stack_data(FLAGS.dir_data_eval)

        # Initialise the variables and run
        init = tf.global_variables_initializer()
        sess.run(init)

        # For train

        saver.restore(sess, "./{0}/checkpoint.ckpt".format(FLAGS.ckpt_name))

        # Training cycle
        summary, acc_train, loss_train, _ = sess.run([merged, accuracy, cost, optimizer], feed_dict=feed_dict(stack_data_train, batch_size, len_input))

        print("Accuracy at step {0}: {1:.5f}".format(epoch, acc_test))



if __name__ == "__main__":
    tf.app.run()