#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages
import os, pickle, time

# Import external packages
import numpy as np
import PIL.Image as Image
import tensorflow as tf

# from params import params
from models import InceptionV1BasedModel

def generate_arr_fpath(dir_src):
	arr_fpath = [os.path.join(dir_src, fname) for fname in os.listdir(dir_src)]
	return arr_fpath

def generate_area_crop(img):
	"""
	args:
		img 		: PIL.Image
	return:
		crop_area 	: tuple, of length = 4
	"""
	size_img = img.size
	min_side = min(size_img)
	padding_h, padding_v= (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
	crop_area = (padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v)
	return crop_area

def generate_arr_rec(arr_fpath, resolution):
	size_1d = resolution[0]*resolution[1]*resolution[2]
	seq_rec = np.zeros(shape=(len(arr_fpath), int(size_1d)), dtype=np.float32)
	for i, fpath in enumerate(arr_fpath):
		img = Image.open(fpath).convert('RGB')
		area_crop = generate_area_crop(img)
		img_crop = img.crop(area_crop)
		img_resize = img_crop.resize(resolution[:2])

		arr_img = np.asarray(img_resize)
		arr1d_img = arr_img.reshape(size_1d)
		seq_rec[i] = arr1d_img  # [1, 0] for normal
	return seq_rec

# Inception-v1
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dir_data_inference", "../../dev-data/project-cmc/pickle/eval",
                       "Directory where the images to infer are located.")
tf.flags.DEFINE_string("ckpt_name", "ckpt", "Name of the checkpoint file")
tf.flags.DEFINE_integer("num_class", 2, "Number of classes. Must match with the number of classes from the model of checkpoint")
tf.flags.DEFINE_integer("first_gpu_id", 0, "ID of the first GPU. Default is 0")
tf.flags.DEFINE_integer("num_gpu", 1, "Number of GPUs to utilise. 1 or even numbers are recommended")
tf.flags.DEFINE_integer("resolution", 448, "Resolution of input images. Must match with the resolution from the model of checkpoint")

# Read pretrained weights
data_saved = {'var_epoch_saved': tf.Variable(0)}

# BUILDING THE COMPUTATIONAL GRAPH
# tf Graph input
len_input = FLAGS.resolution * FLAGS.resolution * 3
num_class = FLAGS.num_class
model = InceptionV1BasedModel(num_class)

num_img = len([fname for fname in os.listdir() if fname.lower().endswith(["bmp","jpg","png","gif"])])

with tf.device("/gpu:{0}".format(FLAGS.first_gpu_id)):
    X = tf.placeholder(tf.float32, [None, len_input])
    # y = tf.placeholder(tf.float32, [None, num_class])

    stack_X = tf.split(X, FLAGS.num_gpu, 0)
    # stack_y = tf.split(y, FLAGS.num_gpu, 0)
    stack_pred = [0] * FLAGS.num_gpu
    stack_deconv = [0] * FLAGS.num_gpu
    # stack_xentropy = [0] * FLAGS.num_gpu
    # stack_cost = [0] * FLAGS.num_gpu
    # stack_grad = [0] * FLAGS.num_gpu
for i in range(FLAGS.num_gpu):
    with tf.device("/gpu:{0}".format(i + FLAGS.first_gpu_id)):
        # Define loss, compute gradients
        stack_pred[i], stack_deconv[i] = model.run(stack_X[i], is_training=False, num_rec=num_img)

with tf.device("/gpu:{0}".format(i + FLAGS.first_gpu_id)):
    # Evaluate model
    pred = tf.concat(stack_pred, axis=0)
    print(pred.get_shape())
    result = tf.argmax(pred)
    print(result.get_shape())
    deconv = tf.concat(stack_deconv, axis=0)
    print(deconv.get_shape())

# RUNNING THE COMPUTATIONAL GRAPH
def main(unused_argv):
    # Define saver
    saver = tf.train.Saver()

    # Configure memory growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Run session
    with tf.Session(config=config) as sess:
        # Initialise the variables and run
        init = tf.global_variables_initializer()
        sess.run(init)

        try:
            saver.restore(sess, "./{0}/checkpoint.ckpt".format(FLAGS.ckpt_name))
            print('Model restored')
        except tf.errors.NotFoundError:
            print('No saved model found')	
        arr_fpath = generate_arr_fpath(FLAGS.dir_data_deploy)
        data_deploy = generate_arr_rec(arr_fpath, (FLAGS.resolution,FLAGS.resolution,3))

        pred_print, result_print, deconv_print = sess.run([pred, result, deconv], feed_dict={X: data_deploy})
        #pred_print, result_print = sess.run([pred, result], feed_dict={X:data_deploy})
        print(pred_print.shape)
        print(deconv_print.shape)
        result_print = np.argmax(pred_print, axis=1)
        for i, fpath in enumerate(arr_fpath):
            print(fpath, result_print[i], pred_print[i])
            print(deconv_print[i].shape)
            dd = deconv_print[i].astype(np.uint8)
            img = Image.fromarray(dd)
            img.save("{0}.bmp".format(i))

if __name__ == "__main__":
    tf.app.run()
