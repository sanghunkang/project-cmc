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
parser.add_argument("-f", "--filename", type=str, help="Prefix to output filenames. \"train.pickle\" and \"validation.pickle\". Default is an empty string.")
parser.add_argument("-r", "--ratio", type=float, help="Ratio between training and validation datasets. Default is 0.8")
parser.add_argument("-p", "--resolution", type=int, help="Number of pixels of both sides. Default is 224")
parser.add_argument("-n", "--sample_size", type=int, help="Number of sample of each class. Default is 1000")
parser.add_argument("-c", "--classlabel", type=str, help="Number of splits is the number of classes, and symbols in each split is classified into that class.")

args = parser.parse_args()

def generate_seq_fpath(dir_src, c):
	dir_src_abs = os.path.join(dir_src, c)
	seq_fpath = os.listdir(dir_src_abs)
	return [os.path.join(dir_src_abs, fpath) for fpath in seq_fpath if fpath.split(".")[-1].lower() in EXT_IMAGE]

def generate_seq_seq_fpath(dir_src, classlabel, sample_size):

	seq_seq_fpath =[]
	for info in classlabel.split("/"):
		seq_fpath = []
		for c in info:
			seq_fpath = seq_fpath + generate_seq_fpath(dir_src, c)
		assert len(seq_fpath)>= sample_size, "Not enough samples"
		seq_seq_fpath.append(seq_fpath[:sample_size])
	return seq_seq_fpath

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

def generate_nparray_from_pilimage(img):
	pass

def generate_seq_rec(seq_fpath, resolution, vec_class):
	size_1d = resolution[0]*resolution[1]*resolution[2]

	seq_rec = np.zeros(shape=(len(seq_fpath), int(size_1d) + len(vec_class)), dtype=np.float32)
	for i, fpath in enumerate(seq_fpath):
		print(fpath)
		img = Image.open(fpath).convert('RGB')
		area_crop = generate_area_crop(img)
		img_crop = img.crop(area_crop)

		# img_crop = img_crop.crop((img_crop.size[0]*0.15, img_crop.size[1]*0, img_crop.size[0]*0.85, img_crop.size[1]*0.70))
		# img_crop.show()


		img_resize = img_crop.resize(resolution[:2])

		arr_img = np.asarray(img_resize)
		arr1d_img = arr_img.reshape(size_1d)
		seq_rec[i] = np.append(arr1d_img, vec_class)  # [1, 0] for normal

	return seq_rec
# Automatically crops into a square. Default is (224,224,3), which is the most prevalent choice of ILSVRC submissions.



def serialize_dir(seq_seq_fpath):
	seq_seq_rec = []
	for i, seq_fpath in enumerate(seq_seq_fpath):
		vec_class = [0 for i in range(len(seq_seq_fpath))]
		vec_class[i] = 1
		seq_seq_rec.append(generate_seq_rec(seq_fpath, resolution, vec_class))
	return seq_seq_rec

def divide_trainNtest(seq_seq_rec, ratio):
	stack_seq_rec_train = []
	stack_seq_rec_test = []
	for seq_rec in seq_seq_rec:
		np.random.shuffle(seq_rec)
		stack_seq_rec_train.append(seq_rec[:int(seq_rec.shape[0]*ratio)])
		stack_seq_rec_test.append(seq_rec[int(seq_rec.shape[0]*ratio):])
	seq_rec_train = np.concatenate((stack_seq_rec_train))
	seq_rec_test = np.concatenate((stack_seq_rec_test))
	return seq_rec_train, seq_rec_test

def write_pickles(seq_seq_rec, dir_dst, prefix_fname):
	seq_rec_train, seq_rec_test = divide_trainNtest(seq_seq_rec, ratio_train2val)
	with open(os.path.join(dir_dst, prefix_fname + "_train.pickle"), 'wb') as handle:
		pickle.dump(seq_rec_train, handle, protocol=2)

	with open(os.path.join(dir_dst, prefix_fname + "_validation.pickle"), 'wb') as handle:
		pickle.dump(seq_rec_test, handle, protocol=2)

dir_src = "./"
dir_dst = "./"
prefix_fname = ""
ratio_train2val = 0.8
resolution = (224,224,3)
sample_size = 1000
classlabel = "0/1"

print(ratio_train2val)

EXT_IMAGE = ["bmp","jpg","png","gif"]

print(args.classlabel)

if args.src: dir_src = args.src
if args.dst: dir_dst = args.dst
if args.filename: prefix_fname = args.filename
if args.ratio: ratio_train2val = args.ratio
if args.resolution: resolution = (args.resolution, args.resolution, 3)
if args.sample_size: sample_size = args.sample_size
if args.classlabel: classlabel = args.classlabel

seq_seq_fpath = generate_seq_seq_fpath(dir_src, classlabel, sample_size)
seq_seq_rec = serialize_dir(seq_seq_fpath)
write_pickles(seq_seq_rec, dir_dst, prefix_fname)

