#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import built-in packages
import sys, os, pickle
from collections import Counter

# Import external packages
from scipy import stats
from skimage.color import rgb2gray, gray2rgb
import skimage

import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

dirpath = "..\\pickle"

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

def add_random_noise(img, mode, **kwargs):
	img_modified = rgb2gray(img / 255)
	img_modified = skimage.util.random_noise(img_modified, mode=mode, **kwargs)#'gaussian')
	img_modified = gray2rgb(img_modified)
	return img_modified

stack_data_raw = generate_stack_data(dirpath)

for data_raw in stack_data_raw:
	print(data_raw.shape)
	# for row in data_raw[:3]:
	arr_img = data_raw[0][:-3]
	img = arr_img.reshape((448,448,3))

	# img_sp = noisy("sp", img_unichannel)

    #
	# img_poisson = noisy("poisson", img_unichannel)
	# img_speckle = noisy("speckle", img_unichannel)
	# plt.imshow(img_gauss)
	# plt.show()

	# Plots
	plt.figure(figsize=(12, 16))

	plt.subplot(231)
	plt.imshow(add_random_noise(img, mode="gaussian", mean=0, var=0), cmap='gray')
	plt.axis('off')


	plt.subplot(232)
	plt.imshow(add_random_noise(img, mode="poisson"), cmap='gray')
	plt.axis('off')

	plt.subplot(233)
	plt.imshow(add_random_noise(img, mode="speckle"), cmap='gray')
	plt.axis('off')

	plt.subplot(235)
	plt.imshow(add_random_noise(img, mode="gaussian", var=0.01), cmap='gray')
	plt.axis('off')

	plt.subplot(236)
	# plt.imshow(add_random_noise(img, mode="localvar", local_vars=np.random.uniform(0, 0.01, (img.shape[0], img.shape[1]))), cmap='gray')
	plt.imshow(add_random_noise(img, mode="localvar", local_vars=stats.truncnorm.rvs(0, 0.05, size=(img.shape[0], img.shape[1]))), cmap='gray')
	plt.axis('off')

	plt.tight_layout()
	plt.show()

