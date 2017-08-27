#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import built-in packages
import os, subprocess, sys, shutil

# Define constants (mostly paths of directories)
DIR_DATA_LABEL = "C:\\dev-data\\labels\\"
DIR_DATA_RAW_LUNG = "C:\\dev-data\\raw-LUNG\\"
DIR_DATA_LABELLED_C00 = "C:\\dev-data\\labelled-C00\\"
DIR_DATA_LABELLED_CNN = "C:\\dev-data\\labelled-CNN\\"

# Check on which system the programme is running
print(sys.platform)

# Open CSV
with open(DIR_DATA_LABEL + "label_LUNG.csv", "r") as fr:
	line = fr.readline()

	# Iterate as long as the given line is good enough
	while len(line)>2:
		line = fr.readline()
		print(line)

		# The element of the 0th position is the fname
		# The element of the 2th position is the size and shape
		fname = line.split(",")[0]
		sizeshape = line.split(",")[1]

		# Define source(i.e. original file) of given image file
		fpath_src = DIR_DATA_RAW_LUNG + sizeshape  + "\\" + fname.replace("-","_") + ".JPG"
		
		# Define destination path of given image file
		# Classify as normal - C00 - if sizeshape is 00, as abnormal otherwise
		# Images of normal case go to directory ending with C00, others go to CNN (for now)
		# ... plus trivial revision of filename
		if sizeshape in ["00"]:
			fpath_dst = DIR_DATA_LABELLED_C00 + "C00_" + fname.replace("-","") + ".JPG"
		else:
			fpath_dst = DIR_DATA_LABELLED_CNN + "C" + sizeshape + "_" + fname.replace("-","") + ".JPG"

		print(fpath_src)
		print(fpath_dst)

		# Copy fpath_src to fpath_dst
		shutil.copyfile(fpath_src, fpath_dst)
