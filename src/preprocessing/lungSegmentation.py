#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import built-in packages
import sys
from collections import Counter

# Import external packages
from scipy import ndimage
from skimage import measure, filters

import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

DIR_DATA_LIGHT = ""
DIR_DATA_LUNG = "C:\\dev-data\\LUNG\\01\\"
fpath = "..\\..\\data_light\\bmp\\I0000001.BMP"

# Define some functions
def make_seq_comp_canddt(img_bylabel, num_canddt):
	# Finds all unique elements and their positions
	unique, pos = np.unique(img_bylabel, return_inverse=True) 

	# Count the number of each unique element
	counts = np.bincount(pos)

	# Select N most frequent labels
	seq_label = np.argpartition(counts, -num_canddt)[-num_canddt:] 

	seq_comp_canddt = []
	for label in seq_label[:-1]: # Neglect the most frequent element, which is the background - i.e. 0
		# Make image (2D matrix) of zeros
		ret = np.zeros(shape=img_bylabel.shape, dtype=np.int32)
		
		# 1 if an element of img_bylabel matches with the label, remain 0 otherwise 
		ret[img_bylabel == label] = 1
		seq_comp_canddt.append(ret)
	return seq_comp_canddt

def make_seq_pair_distComp(seq_comp_canddt, img):
	# Calculate the centroid(1) of the entire image + some shift
	centroid_img = (img.shape[0]/2 - 400, img.shape[1]/2)

	# Sequence of mappings to return
	seq_pair_distComp = []
	for comp_canddt in seq_comp_canddt:
		# Calculate the centroid(2) of all non-zeros
		centroid_comp = ndimage.measurements.center_of_mass(comp_canddt)

		# Calculate Euclidean distance between the two centorids
		dist2center = ((centroid_img[0] - centroid_comp[0])**2 + (centroid_img[1] - centroid_comp[1])**2)**0.5
		
		# Map the distance to the component information, append it to the sequence to return
		pair_distComp = [dist2center, comp_canddt]
		seq_pair_distComp.append(pair_distComp)
	
	return seq_pair_distComp

def make_section(seq_pair_distComp):
	# Sort by ascending order, so that the centermost elements be located at the foremost entries
	seq_pair_distComp.sort()

	# Get the two centermost elements and overlap them
	comp0 = seq_pair_distComp[0][1]
	comp1 = seq_pair_distComp[1][1]
	section_lung = comp0 + comp1
	
	return section_lung

##############################################################################
# Step 1: Calculate threshhold and convert into binary image

# Read data
img = cv2.imread(fpath ,0)

# Otsu's thresholding after Gaussian filtering
img_blur = cv2.GaussianBlur(img, (5,5), 0)
_, th_otsugauss = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU) 

# Rescale to {0, 1}
th_otsugauss = (th_otsugauss/255-1)*(-1)

# Set the dtype into np.int32
kernel_ccl = np.asarray(th_otsugauss, dtype=np.int32)

##############################################################################
# Step 2: Connected Component Labelling

# Label images by connected component
img_bylabel = measure.label(kernel_ccl)

# 
seq_comp_canddt = make_seq_comp_canddt(img_bylabel, 10)

# Add distance information to each component
seq_pair_distComp = make_seq_pair_distComp(seq_comp_canddt, img)

# Make section
section = make_section(seq_pair_distComp) 

##############################################################################
# Step 3: Morphological Reconstruction

kernel_mr = np.ones((10,10), np.int32) # define structure for morphological reconstruction
section_mr = ndimage.binary_dilation(section, structure=kernel_mr, iterations=10) # dilation
section_mr = ndimage.binary_erosion(section_mr, structure=kernel_mr, iterations=5) # erosion
img_final = section_mr*img

##############################################################################
# Plots

plt.figure(figsize=(12,8))

plt.subplot(231)
plt.imshow(kernel_ccl, cmap='gray')
plt.axis('off')

plt.subplot(232)
plt.imshow(img_bylabel, cmap='nipy_spectral')
plt.axis('off')

plt.subplot(233)
plt.imshow(section, cmap='gray')
plt.axis('off')

plt.subplot(234)
plt.imshow(section_mr, cmap='gray')
plt.axis('off')

plt.subplot(235)
plt.imshow(img_final, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()