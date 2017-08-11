#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import os, sys, shutil

# Define constants (mostly paths of directories)
DIR_DATA_LABELLED_CNN = "C:\\dev-data\\labelled-CNN-dcm"

DIR_DATA_LABELLED_C0N = "C:\\dev-data\\labelled-C0N"
DIR_DATA_LABELLED_C1N = "C:\\dev-data\\labelled-C1N"
DIR_DATA_LABELLED_C2N = "C:\\dev-data\\labelled-C2N"
DIR_DATA_LABELLED_C3N = "C:\\dev-data\\labelled-C3N"


# Check on which system the programme is running
print(sys.platform)


def check_dir(dirpath):
    if not os.path.isdir(dirpath): os.mkdir(dirpath)

check_dir(DIR_DATA_LABELLED_C0N)
check_dir(DIR_DATA_LABELLED_C1N)
check_dir(DIR_DATA_LABELLED_C2N)
check_dir(DIR_DATA_LABELLED_C3N)

for fpath in os.listdir(DIR_DATA_LABELLED_CNN):
    fpath_src = os.path.join(DIR_DATA_LABELLED_CNN, fpath)

    DIR_DATA_LABELLED_CXN = "C:/dev-data/labelled-C{0}N".format(fpath[1])
    fpath_dir = os.path.join(DIR_DATA_LABELLED_CXN, fpath)

    # if fpath[1] == "0": fpath_dir = os.path.join(DIR_DATA_LABELLED_C0N, fpath)
    # elif fpath[1] == "1": fpath_dir = os.path.join(DIR_DATA_LABELLED_C1N, fpath)
    # elif fpath[1] == "2": fpath_dir = os.path.join(DIR_DATA_LABELLED_C2N, fpath)
    # elif fpath[1] == "3": fpath_dir = os.path.join(DIR_DATA_LABELLED_C3N, fpath)

    print(fpath_dir)
    shutil.copy(fpath_src, fpath_dir)