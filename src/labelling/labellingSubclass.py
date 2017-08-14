#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import os, sys, shutil

# Define constants (mostly paths of directories)
DIR_DATA_LABELLED_CNN = "C:\\dev-data\\labelled_tmp"

# Check on which system the programme is running
print(sys.platform)

def check_dir(dirpath):
    if not os.path.isdir(dirpath): os.mkdir(dirpath)

def copy_from_src2dst(fpath):
    fpath_src = os.path.join(DIR_DATA_LABELLED_CNN, fpath)

    DIR_DATA_LABELLED_CXN = "C:\\dev-data\\labelled-C{0}N".format(fpath[1])
    fpath_dir = os.path.join(DIR_DATA_LABELLED_CXN, fpath)

    print(fpath_dir)
    shutil.copy(fpath_src, fpath_dir)

for i in range(4): check_dir("C:\\dev-data\\labelled-C{0}N".format(str(i)))
for fpath in os.listdir(DIR_DATA_LABELLED_CNN): copy_from_src2dst(fpath)