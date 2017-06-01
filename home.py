#!/usr/bin/python
# -*- coding: utf-8 -*-

# Import 3rd-party packages
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

# def open_image(path_img):
# 	return Image.open(path_img)

# def crop_innersqr(img_open):
# 	size_img = img_open.size
# 	min_side = min(size_img)
# 	padding_h, padding_v = (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
# 	img_open.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))
# 	return img_open

# def reshape_image(img_open, shape):
# 	ret = img_open.resize(shape) 
# 	return ret

# def serialize_image(img_open):
# 	img_serialized = np.asarray(img_open)
# 	ret = img_serialized.reshape(1, img_serialized.size)[0]
# 	return ret

# def process_image(path_img, resolution):
# 	img_open = Image.open(path_img)
# 	img_cropped = crop_innersqr(img_open)
# 	img_reshaped = img_cropped.resize(resolution) 
# 	# img_reshaped = reshape_image(img_cropped, resolution)
# 	return serialize_image(img_reshaped)

path_img = "C:\\dev-data\\project-cucm\\0.png"
# path_img = "/usr/local/dev-data//project-cucm//0.png"

img_open = Image.open(path_img) # Open file, create "image object"
img_open = img_open.convert("L") # Convert colour format into grayscale
arr_img = np.asarray(img_open) # Get numpy-array form image object

# Print intermediate results
print(arr_img.shape)
print(np.amax(arr_img))
print(np.amin(arr_img))
print(arr_img)

# Show image from numpy-array
plt.imshow(arr_img)
plt.show()

# Create an all 130 numpy-array, same shape with arr_img
arr_th = np.full(arr_img.shape, 130) 

# Create a True/False numpy-array
# True if an element of arr_img of corresponding position is less than 130, False otherwise
arr_img_truefalse = np.less(arr_img, arr_th) 

# Elementwise multiplication of the original array and True/False array
arr_img_selected = np.multiply(arr_img_truefalse, arr_img)

# Print intermediate results
print(arr_img_selected.shape)
print(np.amax(arr_img_selected))
print(np.amin(arr_img_selected))

# Show image from numpy-array
plt.imshow(arr_img_selected)
plt.show()