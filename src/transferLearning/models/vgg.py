# from kaffe.tensorflow import Network

# class VGG16(Network):
#     def setup(self):
#         (self.feed('data')
#              .conv(3, 3, 64, 1, 1, name='conv1_1')
#              .conv(3, 3, 64, 1, 1, name='conv1_2')
#              .max_pool(2, 2, 2, 2, name='pool1')
#              .conv(3, 3, 128, 1, 1, name='conv2_1')
#              .conv(3, 3, 128, 1, 1, name='conv2_2')
#              .max_pool(2, 2, 2, 2, name='pool2')
#              .conv(3, 3, 256, 1, 1, name='conv3_1')
#              .conv(3, 3, 256, 1, 1, name='conv3_2')
#              .conv(3, 3, 256, 1, 1, name='conv3_3')
#              .max_pool(2, 2, 2, 2, name='pool3')
#              .conv(3, 3, 512, 1, 1, name='conv4_1')
#              .conv(3, 3, 512, 1, 1, name='conv4_2')
#              .conv(3, 3, 512, 1, 1, name='conv4_3')
#              .max_pool(2, 2, 2, 2, name='pool4')
#              .conv(3, 3, 512, 1, 1, name='conv5_1')
#              .conv(3, 3, 512, 1, 1, name='conv5_2')
#              .conv(3, 3, 512, 1, 1, name='conv5_3')
#              .max_pool(2, 2, 2, 2, name='pool5')
#              .fc(4096, name='fc6')
#              .fc(4096, name='fc7')
#              .fc(1000, relu=False, name='fc8')
#              .softmax(name='prob'))

import numpy as np
import tensorflow as tf

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
      # Conv2D wrapper, with bias and relu activation
      x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
      x = tf.nn.bias_add(x, b)
      return tf.nn.relu(x)

def maxpool2d(x, k=2):
      # MaxPool2D wrapper
      return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Create model
def conv_net(x, params):
      # Reshape input picture
      x = tf.reshape(x, shape=[-1, 64, 64, 3])

      # Convolution and max pooling(down-sampling) Layer
      conv11 = conv2d(x, params['conv1_1_W'], params['conv1_1_b'])
      conv12 = conv2d(conv11, params['conv1_2_W'], params['conv1_2_b'])
      conv13 = conv2d(conv12, params['conv1_3_W'], params['conv1_3_b'])
      conv13 = maxpool2d(conv13, k=2)

      # Convolution and max pooling(down-sampling) Layer
      conv21 = conv2d(conv13, params['conv2_1_W'], params['conv2_1_b'])
      conv22 = conv2d(conv21, params['conv2_2_W'], params['conv2_2_b'])
      conv23 = conv2d(conv22, params['conv2_3_W'], params['conv2_3_b'])
      conv23 = maxpool2d(conv23, k=2)

      # Convolution and max pooling(down-sampling) Layer
      conv31 = conv2d(conv23, params['conv3_1_W'], params['conv3_1_b'])
      conv32 = conv2d(conv31, params['conv3_2_W'], params['conv3_2_b'])
      conv33 = conv2d(conv32, params['conv3_3_W'], params['conv3_3_b'])
      conv33 = maxpool2d(conv33, k=2)

      # Convolution and max pooling(down-sampling) Layer
      # conv41 = conv2d(conv33, params['W_conv41'], params['b_conv41'])
      # conv42 = conv2d(conv41, params['W_conv42'], params['b_conv42'])
      # conv43 = conv2d(conv42, params['W_conv43'], params['b_conv43'])
      # conv43 = maxpool2d(conv43, k=2)

      # Reshape conv2 output to fit fully connected layer input
      fc1 = tf.reshape(conv33, [-1, params['fc6_W'].get_shape().as_list()[0]])

      # Fully connected layer and Apply Dropout
      fc1 = tf.add(tf.matmul(fc1, params['fc6_W']), params['fc6_b'])
      fc1 = tf.nn.relu(fc1)
      fc1 = tf.nn.dropout(fc1, params['dropout'])

      # Fully connected layer and Apply Dropout
      fc2 = tf.add(tf.matmul(fc1, params['fc7_W']), params['fc7_b'])
      fc2 = tf.nn.relu(fc2)
      fc2 = tf.nn.dropout(fc2, params['dropout'])

      # Output, class prediction
      out = tf.add(tf.matmul(fc2, params['fc8_W']), params['fc8_b'])
      # out = tf.nn.l2_normalize(out, 1, epsilon=1e-12, name=None)
      return out
