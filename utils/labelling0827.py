#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import argparse, os, sys, shutil

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="Directory where original data are stored.")
parser.add_argument("-d", "--dst", type=str, help="Directory where original data will be copied into its subdirectories according to their classfications.")
parser.add_argument("-r", "--ref", type=str, help="path to .csv file where relevant data is located.")
args = parser.parse_args()

dir_src = "./"
dir_dst = "./"
fpath_ref = ""

if args.src: dir_src = args.src
if args.dst: dir_dst = args.dst
if args.ref: fpath_ref = args.ref

# Check on which system the programme is running
print(sys.platform)

df = pd.read_csv(fpath_ref)

def check_dir(fpath):
    if not os.path.isdir(fpath.rsplit("/", 1)[0]): os.mkdir(fpath.rsplit("/", 1)[0])

arr_fpath_src = [os.path.join(dir_src, str(df["PID"][i]) + ".bmp") for i in range(df.shape[0])]
arr_fpath_dst = [os.path.join(dir_dst, "{}/{}.bmp".format(df["TYPE1"][i][0], df["PID"][i])) for i in range(df.shape[0])]

for fpath_src, fpath_dst in zip(arr_fpath_src, arr_fpath_dst):
	print(fpath_src)
	try:
		check_dir(fpath_dst)
		shutil.copy(fpath_src, fpath_dst)
		print("done")
	except FileNotFoundError:
		print("Not if the reference file")