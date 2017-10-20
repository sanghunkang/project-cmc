#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import built-in packages
import argparse, os, pickle, time

# Import external packages
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def generate_area_crop(img):
	"""
	args:
		img 		: PIL.Image
	return:
		crop_area 	: tuple, of length = 4
	"""
	size_img = img.size
	min_side = min(size_img)
	padding_h, padding_v = (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
	crop_area = (padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v)
	return crop_area

# from params import params
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="not yet written")
args = parser.parse_args()

arr_fpath = os.listdir(args.dir)
data = np.zeros(shape=(len(arr_fpath), 448, 448, 3))

for i, fpath in enumerate(arr_fpath):
	print(fpath)
	img = Image.open(os.path.join(args.dir, fpath)).convert('RGB')
	# area_crop = generate_area_crop(img)
	# img_crop = img.crop(area_crop)
	# img_resize = img_crop.resize((448, 448))
	img_np = np.asarray(img)
	print(img_np)
	data[i] = img_np

avr = np.sum(data, axis=0)/data.shape[0]
stack_avr = np.stack([avr]*data.shape[0])
var = np.sum((data - stack_avr)**2, axis=0)/data.shape[0]
print(avr.shape)
print(var.shape)
plt.imshow(var)
plt.show()