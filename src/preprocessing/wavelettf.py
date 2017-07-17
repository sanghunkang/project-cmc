#!/usr/bin/python
# -*- coding: utf-8 -*-

import math, os, subprocess, sys

# Import external packages
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import pywt

# Open image
# img = Image.open("C:\\dev\\project-cucm\\data_light\\bmp\\I0000001.BMP")
# img = Image.open("C:\\dev\\project-cucm\\data_light\\bmp\\10028041.bmp")
# img = Image.open("C:\\dev\\project-cucm\\data_light\\bmp\\24362776.bmp")
# img = Image.open("C:\\dev\\project-cucm\\data_light\\bmp\\I0000001_crop.BMP")
# img = Image.open("C:\\dev\\project-cucm\\data_light\\bmp\\lena512.BMP")
# img = Image.open("C:\\dev\\project-cucm\\data_light\\bmp\\Lichtenstein.png")

arr_img = np.asarray(img)
arr_img = np.swapaxes(arr_img, 0, 2)
arr_img = np.swapaxes(arr_img, 1, 2)
arr_img = arr_img[0]

plt.imshow(arr_img, 'gray')
plt.show()

_, cA = pywt.dwt(arr_img, 'db2', axis=1)
_, cA = pywt.dwt(arr_img, 'db2', axis=0)

_, cA = pywt.dwt(cA, 'db2', axis=0)
_, cA = pywt.dwt(cA, 'db2', axis=1)

# Show results, people don't believe written results
# cA = LL, cH = LH, cV = HL, cD = HH
LL1, (cH, cV, cD) = pywt.dwt2(arr_img, 'haar')

plt.imshow(LL1, 'gray')
plt.show()

for x in (cH, cV, cD):
	plt.imshow(x, 'gray')
	plt.show()

LL2, (cH, cV, cD) = pywt.dwt2(LL1, 'haar')
plt.imshow(LL2, 'gray')
plt.show()

for x in (cH, cV, cD):
	plt.imshow(x, 'gray')
	plt.show()


LL3, (cH, cV, cD) = pywt.dwt2(LL2, 'haar')
plt.imshow(LL3, 'gray')
plt.show()

for x in (cH, cV, cD):
	plt.imshow(x, 'gray')
	plt.show()



# plt.imshow(cA_H, 'binary')
# plt.show()

# plt.imshow(cA_HV, 'binary')
# plt.show()

# plt.imshow(cA_VH, 'binary')
# plt.show()