#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import sys, shutil

# Define constants (mostly paths of directories)
DIR_DATA_RAW_IMG0705 = "C:\\dev-data\\raw-img0705\\"
DIR_DATA_LABELLED_C00 = "C:\\dev-data\\labelled-C00\\"
DIR_DATA_LABEL = "C:\\dev-data\\labels\\"

# Check on which system the programme is running
print(sys.platform)

# Open CSV
with open(DIR_DATA_LABEL + "label_img0705.csv", "r") as fr:
	line = fr.readline()

	# Iterate as long as the given line is good enough
	while len(line)>2:
		line = fr.readline()

		# The element of the 2nd position is the fname
		fname = line.split(",")[2]
		print(fname)
		
		# Define source(i.e. original file) & destination path of given image file
		fpath_src = DIR_DATA_RAW_IMG0705 + fname + ".dcm"
		fpath_dst = DIR_DATA_LABELLED_C00 + "C00-" + fname + ".dcm"

		# Copy fpath_src to fpath_dst
		shutil.copyfile(fpath_src, fpath_dst)
