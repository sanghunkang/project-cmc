#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os, subprocess, sys, shutil
import _pickle as cPickle
import pickle as pickle

# Import 3rd party packages
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

# Import built-in packages

# Define constants (mostly paths of directories)
DIR_DATA_LABELLED_C00 = "C:\\dev-data\\labelled-C00\\"
DIR_DATA_LABELLED_CNN = "C:\\dev-data\\labelled-CNN\\"
DIR_DATA_PICKLE = "C:\\dev-data\\pickle\\"

len_h = 224*4
len_w = 224*4
n_class = 2 # Normal & abnormal

def dir_to_pickle(dir_src, dir_pickle, resolution, vec_class):
	len_h, len_w = resolution

	seq_fpath_C00 = os.listdir(dir_src)

	seq_train_normal = np.zeros(shape=(len(seq_fpath_C00), len_h*len_w + n_class))
	for i, fpath_C00 in enumerate(seq_fpath_C00):
		print(fpath_C00)
		img = Image.open(dir_src + fpath_C00)
		size_img = img.size
		
		min_side = min(size_img)
		padding_h, padding_v= (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
		
		img_crop = img.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))
		img_resize = img_crop.resize((len_h, len_w))

		arr_img = np.asarray(img_resize)
		if len(arr_img.shape) == 3:
			arr_img = np.swapaxes(arr_img, 0, 2)
			arr_img = np.swapaxes(arr_img, 1, 2)
			arr_img = arr_img[0]

		arr1d_img = arr_img.reshape(len_h*len_w)

		seq_train_normal[i] = np.append(arr1d_img, vec_class) # [1, 0] for normal

	fname = dir_src.split("\\")[-2]
	with open(DIR_DATA_PICKLE + fname + ".pickle", 'wb') as handle:
	    pickle.dump(seq_train_normal, handle, protocol=pickle.HIGHEST_PROTOCOL)

dir_to_pickle(DIR_DATA_LABELLED_C00, DIR_DATA_PICKLE, (len_h, len_w), [1, 0])
dir_to_pickle(DIR_DATA_LABELLED_CNN, DIR_DATA_PICKLE, (len_h, len_w), [0, 1])
