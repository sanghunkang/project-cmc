#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import built-in packages
import sys, os, pickle
from collections import Counter

# Import external packages
from scipy import ndimage
from skimage import measure, filters

import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

dirpath = ".\\data_light\\pickle"

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

# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noisy(noise_typ, image):
	if noise_typ == "gauss":
		row,col= image.shape
		mean = 0
		var = 100
		sigma = var**0.5
		gauss = np.random.normal(mean,sigma,(row,col))
		gauss = gauss.reshape(row,col)
		noisy = image + gauss
		return noisy
	elif noise_typ == "sp":
		# row,col,ch = image.shape
		s_vs_p = 0.5
		amount = 0.01
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
		out[coords] = 0
		return out
	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2 ** np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy
	elif noise_typ =="speckle":
		row,col = image.shape
		gauss = np.random.randn(row,col)
		gauss = gauss.reshape(row,col)
		noisy = image + image * gauss
		return noisy

stack_data_raw = generate_stack_data(dirpath)

for data_raw in stack_data_raw:
	print(data_raw.shape)
	for row in data_raw[:3]:
		arr_img = row[:-2]
		img = arr_img.reshape((448,448,3))

		img_unichannel = img[:,:,0]
		# img_pil = Image.fromarray(img)
		# image_file = image_file.convert('L')
		# plt.imshow(img[:,:,0])
		# plt.show()

		img_gauss = noisy("gauss", img_unichannel)
		img_sp = noisy("sp", img_unichannel)
		img_poisson = noisy("poisson", img_unichannel)
		img_speckle = noisy("speckle", img_unichannel)
		# plt.imshow(img_gauss)
		# plt.show()

		# Plots
		plt.figure(figsize=(12, 16))

		plt.subplot(231)
		plt.imshow(img_unichannel, cmap='gray')
		plt.axis('off')

		plt.subplot(232)
		plt.imshow(img_gauss, cmap='gray')
		plt.axis('off')

		plt.subplot(233)
		plt.imshow(img_sp, cmap='gray')
		plt.axis('off')

		plt.subplot(235)
		plt.imshow(img_poisson, cmap='gray')
		plt.axis('off')

		plt.subplot(236)
		plt.imshow(img_speckle, cmap='gray')
		plt.axis('off')

		plt.tight_layout()
		plt.show()

