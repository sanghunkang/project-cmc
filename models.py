#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages

# Import external packages
import numpy as np
import tensorflow as tf


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


def slice_params_module(name_module, params_pre):
    params_module = {}
    keys = [key for key in params_pre]
    for key in keys:
        if name_module in key:
            params_module[key.replace(name_module, "")] = params_pre[key]
    return params_module


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    # if is_switch == True: return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME'), None
    return tf.nn.max_pool_with_argmax(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv2ser(x, params_fc):
    return tf.reshape(x, [-1, params_fc.get_shape().as_list()[0]])


def fc1d(x, W, b):
    # FC layer wrapper, with bias, relu activation plus batch-normalisation if demanded
    fc = tf.add(tf.matmul(x, W), b)
    return tf.nn.relu(fc)


def inception_module(tsr_X, name_module, params):
    params_module = slice_params_module(name_module, params)
    # 1x1 convolution
    inception_1x1 = conv2d(tsr_X, params_module["1x1_W"], params_module["1x1_b"])

    # 3x3 convolution
    inception_3x3r = conv2d(tsr_X, params_module["3x3_reduce_W"], params_module["3x3_reduce_b"])
    inception_3x3 = conv2d(inception_3x3r, params_module["3x3_W"], params_module["3x3_b"])

    # 5x5 convolution
    inception_5x5r = conv2d(tsr_X, params_module["5x5_reduce_W"], params_module["5x5_reduce_b"])
    inception_5x5 = conv2d(inception_5x5r, params_module["5x5_W"], params_module["5x5_b"])

    # 3x3 max-pool
    inception_pool_proj = conv2d(tsr_X, params_module["pool_proj_W"], params_module["pool_proj_b"])

    # Concatenation of sublayers - i.e. stacking
    inception_concat = tf.concat([inception_1x1, inception_3x3, inception_5x5, inception_pool_proj], axis=-1)
    return inception_concat


class BaseModel(object):
    def __init__(self, num_class):
        raise NotImplementedError

    def run(self, X, is_training):
        raise NotImplementedError


class InceptionV1BasedModel(BaseModel):
    def __init__(self, num_class):
        self.shape = [-1, 448, 448, 3]

        dict_lyr = np.load("../../dev-data/weight-pretrained/googlenet.npy", encoding='latin1').item()  # return dict
        self.params = reformat_params(dict_lyr)
        self.params['fc6_W'] = tf.Variable(tf.random_normal([4096, 4096]), name='fc6_W')
        self.params['fc6_b'] = tf.Variable(tf.random_normal([4096]), name='fc6_b')

        self.params['fc7_W'] = tf.Variable(tf.random_normal([4096, 4096]), name='fc7_W')
        self.params['fc7_b'] = tf.Variable(tf.random_normal([4096]), name='fc7_b')

        self.params['fc8_W'] = tf.Variable(tf.random_normal([4096, num_class]), name='fc8_W')
        self.params['fc8_b'] = tf.Variable(tf.random_normal([num_class]), name='fc8_b')

    def run(self, X, is_training):
        X_reshaped = tf.reshape(X, shape=self.shape)
        # X_reshaped = tf.image.random_contrast(X_reshaped, 0, 1)

        # Convolution and max pooling(down-sampling) Layers
        # Convolution parameters are from pretrained data
        conv1_7x7_s2 = conv2d(X_reshaped, self.params['conv1_7x7_s2_W'], self.params['conv1_7x7_s2_b'])
        conv1_7x7p_s2, switch_conv1 = maxpool2d(conv1_7x7_s2, k=2)

        if is_training == False:
            tf.nn.conv2d_transpose( switch_conv1,
                                    filter=self.params['conv1_7x7_s2_W'],
                                    output_shape=[-1, 448, 448, 3],
                                    strides=[1, strides, strides, 1],
                                    padding='SAME',
                                    data_format='NHWC',
                                    name=None)

        conv2_3x3 = conv2d(conv1_7x7p_s2, self.params['conv2_3x3_W'], self.params['conv2_3x3_b'])
        conv2p_3x3, switch_conv2p_3x3 = maxpool2d(conv2_3x3, k=2)

        # Inception modules
        inception_3a = inception_module(conv2p_3x3, "inception_3a_", self.params)
        inception_3b = inception_module(inception_3a, "inception_3b_", self.params)
        inception_3bp, switch_inception_3bp = maxpool2d(inception_3b, k=2)

        inception_4a = inception_module(inception_3bp, "inception_4a_", self.params)
        inception_4b = inception_module(inception_4a, "inception_4b_", self.params)
        inception_4c = inception_module(inception_4b, "inception_4c_", self.params)
        inception_4d = inception_module(inception_4c, "inception_4d_", self.params)
        inception_4e = inception_module(inception_4d, "inception_4e_", self.params)
        inception_4ep, switch_inception_4ep = maxpool2d(inception_4e, k=2)

        inception_5a = inception_module(inception_4ep, "inception_5a_", self.params)
        inception_5b = inception_module(inception_5a, "inception_5b_", self.params)
        inception_5ap = tf.nn.avg_pool(inception_5b, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')
        print(inception_5ap.get_shape())

        # Fully connected layer, training is done only for here
        fc5 = tf.reshape(inception_5ap, [-1, self.params["fc6_W"].get_shape().as_list()[0]])
        fc5 = tf.contrib.layers.batch_norm(fc5, is_training=is_training, reuse=True)

        fc6 = fc1d(fc5 , self.params["fc6_W"], self.params["fc6_b"])
        fc6 = tf.contrib.layers.batch_norm(fc6, is_training=is_training, reuse=True)

        fc7 = fc1d(fc6, self.params["fc7_W"], self.params["fc7_b"])
        fc7 = tf.contrib.layers.batch_norm(fc7, is_training=is_training, reuse=True)

        pred = fc1d(fc7, self.params["fc8_W"], self.params["fc8_b"])
        return pred


class Vgg16Model(BaseModel):
    def __init__(self, num_class):
        dict_lyr = np.load("../../dev-data/weight-pretrained/googlenet.npy", encoding='latin1').item()  # return dict
        self.params = reformat_params(dict_lyr)
        self.params = {'fc6_W': tf.Variable(tf.random_normal([4096, 4096]), name='fc6_W'),
                       'fc6_b': tf.Variable(tf.random_normal([4096]), name='fc6_b'),

                       'fc7_W': tf.Variable(tf.random_normal([4096, 4096]), name='fc7_W'),
                       'fc7_b': tf.Variable(tf.random_normal([4096]), name='fc7_b'),

                       'fc8_W': tf.Variable(tf.random_normal([4096, num_class]), name='fc8_W'),
                       'fc8_b': tf.Variable(tf.random_normal([num_class]), name='fc8_b')}

    def run(self, X):
        X_reshaped = tf.reshape(X, shape=[-1, 224, 224, 3])

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

        fc6 = tf.reshape(conv5p, [-1, params["fc6_W"].get_shape().as_list()[0]])

        fc6 = tf.add(tf.matmul(fc6, params["fc6_W"]), params["fc6_b"])
        fc6 = tf.nn.relu(fc6)

        fc7 = tf.add(tf.matmul(fc6, params["fc7_W"]), params["fc7_b"])
        fc7 = tf.nn.relu(fc7)

        fc8 = tf.add(tf.matmul(fc7, params["fc8_W"]), params["fc8_b"])
        pred = tf.nn.relu(fc8)
        return pred
