#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# (Hopefully) easy, convenient, system-independent and backward-compatible 
# Pack subdiretories into pickles of numpy.arrays.

# The source code itself was written in Python3, but the output pickle files are compatible with Python2 as well.

# Import built-in modules
import argparse, sys, os, pickle

# Import 3rd party packages
import PIL.Image as Image
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--src", type=str, help="uppermost source directory which contains all subdirectories for dataset")
parser.add_argument("-d", "--dst", type=str, help="uppermost destination directory where the output pickle files will be stored")
parser.add_argument("-f", "--filename", type=str, help="Prefix to output filenames. \"train.pickle\" and \"validation.pickle\". Default is an empty string.")
parser.add_argument("-r", "--ratio", type=float, help="Ratio between training and validation datasets. Default is 0.8")

args = parser.parse_args()

dirpath_src = "./"
dirpath_dst = "./"
prefix_fname = ""
ratio_train2val = 0.8

if args.src: dirpath_src = args.src
if args.dst: dirpath_dst = args.dst
if args.filename: prefix_fname = args.filename
if args.ratio: ratio_train2val = args.ratio

print(ratio_train2val)

EXT_IMAGE = ["bmp","jpg","png","gif"]

def omit_nonimage(seq_fpath):
	seq_fpath_filtered = []
	for fpath in seq_fpath:
		if fpath.split(".")[-1].lower() in EXT_IMAGE:
			seq_fpath_filtered.append()
	return seq_fpath_filtered

def generate_seq_fpath(dir_src):
	seq_fpath = os.listdir(dir_src)
	return omit_nonimage(seq_fpath)

def generate_crop_area(img):
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

def generate_nparray_from_pilimage(img):
	pass


# Automatically crops into a square. Default is (224,224,3), which is the most prevalent choice of ILSVRC submissions.
def dir_to_pickle(dir_src, resolution, vec_class):
	len_h, len_w, len_c = resolution

	# seq_fpath = os.listdir(dir_src)
	seq_fpath = generate_seq_fpath(dir_src)

	seq_rec = np.zeros(shape=(len(seq_fpath), len_h*len_w*len_c + n_class), dtype=np.float32)
	for i, fpath in enumerate(seq_fpath):
		print(fpath)
		img = Image.open(os.path.join(dir_src, fpath)).convert('RGB')
		# img = raw_img.convert('RGB')
		# size_img = img.size
		
		# min_side = min(size_img)
		# padding_h, padding_v= (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
		area_crop = generate_area_crop(img)
		# img_crop = img.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))
		img_crop = img.crop(area_crop)
		
		# img_crop = img_crop.crop((img_crop.size[0]*0.15, img_crop.size[1]*0, img_crop.size[0]*0.85, img_crop.size[1]*0.70))
		# img_crop.show()


		img_resize = img_crop.resize((len_h, len_w))

		arr_img = np.asarray(img_resize)
		arr1d_img = arr_img.reshape(len_h*len_w*len_c)
		seq_rec[i] = np.append(arr1d_img, vec_class) # [1, 0] for normal

	return seq_rec
# sys.argv[1] = 
# 

# Names of directories, which will be classified as 



# The name of directories should be  e.g. 1, 2

with open(os.path.join(dirpath_dst, prefix_fname + "_train.pickle"), 'wb') as handle:
	pickle.dump(seq_rec_train, handle, protocol=2)

with open(os.path.join(dirpath_dst, prefix_fname + "_validation.pickle"), 'wb') as handle:
	pickle.dump(seq_rec_test, handle, protocol=2)
