#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import built-in packages
import argparse, sys, os

# Import external packages
from scipy import ndimage
from skimage import measure, filters

import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

EXT_IMAGE = ["bmp","jpg","png","gif"]

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src", type=str, help="Directory where original data are stored.")
parser.add_argument("-d", "--dst", type=str, help="Directory where original data will be copied into its subdirectories according to their classfications.")
args = parser.parse_args()

def check_dir(fpath):
    if not os.path.isdir(fpath.rsplit("/", 1)[0]): os.mkdir(fpath.rsplit("/", 1)[0])

dir_src = "./"
dir_dst = "./new/"

if args.src: dir_src = args.src
if args.dst: dir_dst = args.dst

arr_fname = [fname for fname in os.listdir(dir_src) if fname.split(".")[-1] in EXT_IMAGE]
for fname in arr_fname:
    print(fname)
    # Read data
    img = cv2.imread(os.path.join(dir_src, fname) ,0)

    # Step 1: Morphological Reconstruction
    # kernel = np.ones((3,3),np.uint8)
    # erosion = cv2.erode(img,kernel,iterations = 1)
    # opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    # dilation = cv2.dilate(opening, kernel,iterations = 1)
    # closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    # img = closing

    # Step 2: Calculate threshhold and convert into binary image
    # Otsu's thresholding after Gaussian filtering
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    _, th_otsugauss = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)

    # Rescale to {0, 1}
    th_otsugauss = (th_otsugauss/255-1)*(-1)

    # Set the dtype into np.int32
    kernel_ccl = np.asarray(th_otsugauss, dtype=np.int32)
    img_final = img*kernel_ccl
    # Step 3: Connected Component Labelling

    # Label images by connected component

    # Make section
    # Step 4: Give paddings

    # kernel_mr = np.ones((10,10), np.int32) # define structure for morphological reconstruction
    # section_mr = ndimage.binary_dilation(section, structure=kernel_mr, iterations=10) # dilation
    # section_mr = ndimage.binary_erosion(section_mr, structure=kernel_mr, iterations=5) # erosion
    # img_final = section_mr*img
    # img_final = section*img

    # Plots
    plt.figure(figsize=(12,4))

    plt.subplot(131)
    plt.imshow(kernel_ccl, cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(img, cmap='nipy_spectral')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(img_final, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    result = Image.fromarray(img_final.astype(np.uint8))
    result.save(os.path.join(dir_dst, fname))