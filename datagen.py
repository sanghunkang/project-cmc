#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# (Hopefully) easy, convenient, system-independent and backward-compatible 
# Pack subdiretories into pickles of numpy.arrays.

# The source code itself was written in Python3, but the output pickle files are compatible with Python2 as well.

# Import built-in modules
import argparse, os, pickle, random

# Import 3rd party packages
import PIL.Image as Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="uppermost source directory which contains all subdirectories for dataset")
parser.add_argument("-d", "--dst", type=str, help="uppermost destination directory where the output pickle files will be stored")
parser.add_argument("-f", "--filename", type=str, help="Output file name")
parser.add_argument("-r", "--resolution", type=int, help="Number of pixels of both sides. Default is 224")
parser.add_argument("-c", "--classlabels", type=str, help="Number of splits is the number of classes, and symbols in each split is classified into that class.")
args = parser.parse_args()

def generate_arr_fpath(dir_src):
	arr_fpath = [os.path.join(dir_src, fname) for fname in os.listdir(dir_src)]
	return arr_fpath

def reformat_classlabel(fpath):
	fname = fpath.split("/")[-1]
	# Labelling by confirmed diagnosis
	if fname[0] != "X":
		if fname[0] == "S": classlabel = "0"
		elif fname[0] == "0": classlabel = "N"
		else: classlabel = fname[0]
	# Labelling by temporary diagnosis
	elif fname[0] == "X":
		if fname[2:4] == "00": classlabel = "N"
		else: classlabel = fname[2]
	return classlabel

def filter_arr_fpath(arr_fpath, classlabel):
	"""
	args:
		arr_fpath 			:
		classlabel 			: str
	return:
		arr_fpath_filtered 	:
	"""
	arr_fpath_filtered = []
	for fpath in arr_fpath:
		classlabel_filename = reformat_classlabel(fpath)
		if classlabel_filename == classlabel:
			arr_fpath_filtered.append(fpath)
			print("Added to array : {0} {1}".format(classlabel_filename, fpath))
		else:
			print("Irrelevant file: {0} {1}".format(classlabel_filename, fpath))
	return arr_fpath_filtered

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

def generate_arr_rec(arr_fpath, resolution, vec_class):
	size_1d = resolution[0]*resolution[1]*resolution[2]
	seq_rec = np.zeros(shape=(len(arr_fpath), int(size_1d) + len(vec_class)), dtype=np.float32)
	for i, fpath in enumerate(arr_fpath):
		print(fpath)
		img = Image.open(fpath).convert('RGB')
		area_crop = generate_area_crop(img)
		img_crop = img.crop(area_crop)
		# img_crop = img_crop.crop((img_crop.size[0]*0.15, img_crop.size[1]*0, img_crop.size[0]*0.85, img_crop.size[1]*0.70))
		img_resize = img_crop.resize(resolution[:2])

		arr_img = np.asarray(img_resize)
		arr1d_img = arr_img.reshape(size_1d)
		seq_rec[i] = np.append(arr1d_img, vec_class)  # [1, 0] for normal
	return seq_rec

def write_pickles(arr_rec, dir_dst, prefix_fname, classlabel):
	with open(os.path.join(dir_dst, "{0}_{1}.pickle".format(prefix_fname, classlabel)), 'wb') as handle:
		pickle.dump(arr_rec, handle, protocol=2)

dir_src = "./"
dir_dst = "./"
prefix_fname = ""
resolution = (224,224,3)
sample_size = 1000
classlabels = "N0123"
is_balanced = True

EXT_IMAGE = ["bmp","jpg","png","gif"]

if args.src: dir_src = args.src
if args.dst: dir_dst = args.dst
if args.filename: prefix_fname = args.filename
if args.resolution: resolution = (args.resolution, args.resolution, 3)
if args.classlabels: classlabels = args.classlabels

for i, classlabel in enumerate(classlabels):
	print(classlabel)
	arr_fpath = generate_arr_fpath(dir_src)
	arr_fpath = filter_arr_fpath(arr_fpath, classlabel)
	vec_class = [0]*len(classlabels)
	vec_class[i] = 1
	print(vec_class)
	arr_rec = generate_arr_rec(arr_fpath, resolution, vec_class)
	print(arr_rec.shape)
	write_pickles(arr_rec, dir_dst, prefix_fname, classlabel)

# seq_seq_fpath = generate_seq_seq_fpath(dir_src, classlabel, sample_size, is_balanced)
# seq_seq_rec = serialize_dir(seq_seq_fpath)
# write_pickles(seq_seq_rec, dir_dst, prefix_fname)
