#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import 3rd-party packages
import math, os, subprocess, sys

import numpy as np
from skimage import io
from skimage import transform as tf

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



# Load the image as a matrix
image = io.imread("C:\\dev\\project-cucm\\data_light\\bmp\\I0000001.BMP")

# Create Afine transform
tform_aff = tf.AffineTransform(shear=0.2)
modified_aff = tf.warp(image, tform_aff)

io.imshow(modified_aff)
print(image.shape)
# io.show()
io.imsave("C:\\dev\\project-cucm\\data_light\\bmp\\I0000001_0.BMP", image)