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

df = pd.read_csv(fpath_ref, delimiter=',', encoding="utf-8-sig")

def check_dir(fpath):
    if not os.path.isdir(fpath.rsplit("/", 1)[0]): os.mkdir(fpath.rsplit("/", 1)[0])

for i in range(len(df)):
    try:
        fpath_src = os.path.join(dir_src, str(df["PID"][i]) + ".bmp")
        if type(df["TYPE2"][i]) in [int, str]:
            fname = "{0}_{1}{2}_{3}.bmp".format(df["TYPE2"][i], str(df["TYPE1"][i][0]), str(df["TYPE1"][i][2]), df["PID"][i])
        else:
            fname = "{0}_{1}{2}_{3}.bmp".format("X", str(df["TYPE1"][i][0]), str(df["TYPE1"][i][2]), df["PID"][i])
        fpath_dst = os.path.join(dir_dst, fname)
        print(fpath_dst)
        # check_dir(fpath_dst)
        shutil.copy(fpath_src, fpath_dst)
        print(fname)
    except FileNotFoundError as e:
        print(e)
        # print("FileNotFoundError at {0}".format(str(df["PID"][i])))