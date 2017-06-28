#!/usr/bin/python
# -*- coding: utf-8 -*-
#영상번호,병형L,병형R,SHAPE,SIZE,,,,,,,,,,,
# Import external party packages
import os, subprocess, sys, shutil

DIR_DATA_LIGHT = ""
DIR_DATA_LUNG = "C:\\dev-data\\LUNG\\"
DIR_DATA_LUNG_SUM = "C:\\dev-data\\LUNG\\SUM\\"
# fname = "KoSDI_008.JPG"
# fpath = DIR_DATA_LUNG + fname
# fpath = "C:\\dev\\project-cucm\\data_light\\bmp\\I0000001.BMP"

print(sys.platform)

seq_subdir = []
# for subdir in os.listdir(DIR_DATA_LUNG):
# 	if len(subdir) < 5: seq_subdir.append(DIR_DATA_LUNG + subdir + "\\")

# for subdir in seq_subdir: print(subdir)

with open(DIR_DATA_LUNG + "labelling.txt", "r") as fr:
	line = fr.readline()

	while len(line)>2:
		line = fr.readline()

		fname = line.split(",")[0]
		char_l = line.split(",")[1]
		char_r = line.split(",")[2]
		shape = line.split(",")[3]
		size = line.split(",")[4]
		
		fpath_src = DIR_DATA_LUNG + char_l + char_r  + "\\" + fname.replace("-","_") + ".JPG"
		fpath_dst = DIR_DATA_LUNG_SUM + fname  + "_" + char_l + char_r  + "_" + shape + size + ".JPG"
		print(fpath_src)
		print(fpath_dst)

		shutil.copyfile(fpath_src, fpath_dst)
