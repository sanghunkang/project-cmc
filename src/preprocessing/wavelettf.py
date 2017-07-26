#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, os, subprocess, sys

# Import external packages
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import pywt

# Directory to process
DIR_FPATH = "../../data_light/"

def convert_img2arr(fpath):
	img = Image.open(fpath)
	arr_img = np.asarray(img)

	if len(arr_img.shape) == 3:
		arr_img = np.swapaxes(arr_img, 0, 2)
		arr_img = np.swapaxes(arr_img, 1, 2)
		arr_img = arr_img[0]

	return arr_img

# Show results, people don't believe written results
def transform_wavelet(arr_img, times):
	seq_E_kl = []
	for i in range(times):
		LL_N, (LH_N, HL_N, HH_N) = pywt.dwt2(arr_img, 'haar')
		seq_img_subband = [LL_N, LH_N, HL_N, HH_N]
		for img_subband in seq_img_subband:
			E_kl = np.sum(img_subband**2/img_subband.shape[0]/img_subband.shape[1])
			# print(E_kl)
			seq_E_kl.append(E_kl)
		arr_img = LL_N

		# plt.imshow(LL_N, 'gray')
		# plt.show()
	return seq_E_kl

def reformat_seq2csvstr(seq):
	str_csv = ""
	for E_kl in seq: str_csv = str_csv + str(E_kl) + ", "
	str_csv = str_csv.strip(", ")
	return str_csv

str_fwrite = ""
for fpath in os.listdir(DIR_FPATH):
	try:
		fpath_abs = DIR_FPATH + fpath
		arr_img = convert_img2arr(fpath_abs)
		seq_E_kl = transform_wavelet(arr_img, 7)
		str_E_kl = reformat_seq2csvstr(seq_E_kl)
		print(str_E_kl)

		str_fwrite = str_fwrite + str_E_kl + "\n"
	except IsADirectoryError:
		pass

with open("wt.csv", "w", encoding="utf-8") as fw:
	fw.write(str_fwrite)