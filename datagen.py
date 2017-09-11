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
parser.add_argument("-c", "--classlabel", type=str, help="Number of splits is the number of classes, and symbols in each split is classified into that class.")
args = parser.parse_args()

def generate_arr_fpath(dir_src):
	arr_fpath = [os.path.join(dir_src, fname) for fname in os.listdir(dir_src)]
	return arr_fpath

def reformat_classlabel(fpath):
	fname = arr_fpath.split("/")[-1]
	# Labelling by confirmed diagnosis
	if fname[0] != "X":
		if fname[0] != "S": classlabel = "0"
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
			print("Added to array : {0}".format(fpath))
		else:
			print("Irrelevant file: {0}".format(fpath))
	return arr_fpath_filtered

def write_pickles(seq_seq_rec, dir_dst, prefix_fname):	
	with open(os.path.join(dir_dst, prefix_fname + ".pickle"), 'wb') as handle:
		pickle.dump(seq_rec_train, handle, protocol=2)

dir_src = "./"
dir_dst = "./"
prefix_fname = ""
resolution = (224,224,3)
sample_size = 1000
classlabel = "N"
is_balanced = True

EXT_IMAGE = ["bmp","jpg","png","gif"]

if args.src: dir_src = args.src
if args.dst: dir_dst = args.dst
if args.filename: prefix_fname = args.filename
if args.resolution: resolution = (args.resolution, args.resolution, 3)
if args.classlabel: classlabel = args.classlabel

arr_fpath = generate_arr_fpath(dir_src)
arr_fpath filter_arr_fpath(arr_fpath, classlabel)

# seq_seq_fpath = generate_seq_seq_fpath(dir_src, classlabel, sample_size, is_balanced)
# seq_seq_rec = serialize_dir(seq_seq_fpath)
# write_pickles(seq_seq_rec, dir_dst, prefix_fname)