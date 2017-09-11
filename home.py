#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import argparse, os, sys, shutil

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="Directory where original data are stored.")
parser.add_argument("-d", "--dst", type=str, help="Directory where original data will be copied into its subdirectories according to their classfications.")
args = parser.parse_args()

dir_src = "./"
dir_dst = "./"

if args.src: dir_src = args.src
if args.dst: dir_dst = args.dst

# Check on which system the programme is running
print(sys.platform)

for fname in os.listdir(dir_src):
	fpath_src = os.path.join(dir_src, fname)
	fname_final = "X_00_{0}".format(fname)
	fpath_dst = os.path.join(dir_dst, fname_final)
	print(fpath_dst)
	# check_dir(fpath_dst)
	shutil.copy(fpath_src, fpath_dst)
	print(fname)