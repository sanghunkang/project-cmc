#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import 3rd-party packages
import math, os, subprocess, sys

import numpy as np
from skimage import io
from skimage import data
from skimage import transform
from skimage.transform import swirl

import matplotlib.pyplot as plt

print(sys.platform)

# if "linux" in sys.platform:
# 	DIR_DATA = "/usr/local/dev/project-cucm/data_light/"

# 	# convert /usr/local/dev/project-cucm/data_light/I0000001 /usr/local/dev/project-cucm/data_light/I0000001.BMP
# 	cmd = "convert " + DIR_DATA + "I0000001 " + DIR_DATA + "I0000001.BMP"
# 	subprocess.call([cmd], shell=True)
# 	# subprocess.call(["convert " + DIR_DATA + "I0000002 " + DIR_DATA + "I0000002.BMP"], shell=True)
# elif "win" in sys.platform:
# 	DIR_DATA = "C:\\dev\\project-cucm\\data_light\\"
# 	cmd = "convert " + DIR_DATA + "I0000001 " + DIR_DATA + "I0000001.BMP"
# 	print(cmd)
# 	subprocess.call([cmd], shell=True)

fpath = "..\\..\\data_light\\bmp\\I0000001.BMP"

# Load the image as a matrix
image = io.imread(fpath)

# Create Afine transform
# tform_aff = transform.AffineTransform(shear=0.2)
# modified_aff = transform.warp(image, tform_aff)

# io.imshow(modified_aff)
# print(image.shape)
# # io.show()
# io.imsave(fpath, image)

# image = data.checkerboard()
swirled = swirl(image, center=(1000,1000), rotation=0, strength=1, radius=1500)
swirled = swirl(image, center=(2000,2000), rotation=0, strength=1, radius=1500)

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                               sharex=True, sharey=True,
                               subplot_kw={'adjustable':'box-forced'})

ax0.imshow(image, cmap=plt.cm.gray, interpolation='none')
ax0.axis('off')
ax1.imshow(swirled, cmap=plt.cm.gray, interpolation='none')
ax1.axis('off')

plt.show()