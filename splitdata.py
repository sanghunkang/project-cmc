#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# (Hopefully) easy, convenient, system-independent and backward-compatible
# Pack subdiretories into pickles of numpy.arrays.

# The source code itself was written in Python3, but the output pickle files are compatible with Python2 as well.

# Import built-in modules
import argparse, os, pickle, sys

# Import 3rd party packages
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fpath", type=str, help="fpath of data to split")
parser.add_argument("-r", "--ratio", type=float, help="ratio of train set")
parser.add_argument("-d", "--dir_dst", type=str, help="fpath of data to split")
args = parser.parse_args()

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

def get_fname(fpath):
	if "win" in sys.platform: return fpath.split("\\")[-1]
	elif "linux" in sys.platform: return fpath.split("/")[-1]

if args.fpath: fpath = args.fpath
if args.ratio: ratio = args.ratio
if args.dir_dst: dir_dst = args.dir_dst

data = read_data(fpath)
idx_split = int(data.shape[0]*ratio)
print(data.shape, idx_split)

data_train = data[:idx_split]
data_eval = data[idx_split:]
print(data_train.shape)
print(data_eval.shape)

with open(os.path.join(dir_dst, "{0}_train.pickle".format(get_fname(fpath))), 'wb') as handle:
	pickle.dump(data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(dir_dst, "{0}_eval.pickle".format(get_fname(fpath))), 'wb') as handle:
	pickle.dump(data_eval, handle, protocol=pickle.HIGHEST_PROTOCOL)