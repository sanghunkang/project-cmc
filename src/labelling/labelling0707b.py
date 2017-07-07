#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import sys, shutil

# Define constants (mostly paths of directories)
DIR_DATA_RAW_IMG0707B = "C:\\dev-data\\raw-img0707b\\"
DIR_DATA_LABELLED_CNN_DCM = "C:\\dev-data\\labelled-CNN-dcm\\"
DIR_DATA_LABEL = "C:\\dev-data\\labels\\"

# Check on which system the programme is running
print(sys.platform)
# Open CSV
with open(DIR_DATA_LABEL + "label_img0707b.csv", "r") as fr:
	line = fr.readline()

	# Iterate as long as the given line is good enough
	while len(line)>2:
		line = fr.readline()

		# The element of the 2nd position is the fname
		fname = line.split(",")[0]
		cname = line.split(",")[1].strip("\n")
		print(fname)
		
		# Define source(i.e. original file) & destination path of given image file
		fpath_src = DIR_DATA_RAW_IMG0707B + fname + ".dcm"
		fpath_dst = DIR_DATA_LABELLED_CNN_DCM + "C" + cname + "-" + fname + "b.dcm"

		# Copy fpath_src to fpath_dst
		try:
			shutil.copyfile(fpath_src, fpath_dst)
			print(fname)
		except FileNotFoundError:
			print(fname + " MISSING!")