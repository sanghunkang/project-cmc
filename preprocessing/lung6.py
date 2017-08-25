#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import built-in packages
import sys, os
from collections import Counter

# Import external packages
from scipy import ndimage
from skimage import measure, filters

import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

FLAG_MODE = sys.argv[1]

if FLAG_MODE == "-d":
    PATH_SRC = sys.argv[2]
    PATH_DST = sys.argv[3]
elif FLAG_MODE == "-f":
    PATH_SRC = sys.argv[2]
    PATH_DST = sys.argv[3]

# FLAG_MODE = "-f"
# PATH_SRC = "../../data_light/bmp/2.bmp"
# PATH_DST = "../../data_light/sample.bmp"

# Define some functions
def make_seq_fpath(FLAG_MODE, PATH_SRC, PATH_DST=None):
    seq_fpath_src = []
    seq_fpath_dst = []

    if FLAG_MODE == "-f":
        seq_fpath_src.append(PATH_SRC)
        seq_fpath_dst.append(PATH_DST)
    elif FLAG_MODE == "-d":
        for fname in os.listdir(PATH_SRC):
            seq_fpath_src.append(os.path.join(PATH_SRC, fname))
            seq_fpath_dst.append(os.path.join(PATH_DST, fname))
    return zip(seq_fpath_src, seq_fpath_dst)


def make_seq_comp_canddt(img_bylabel, num_canddt, size_min):
    # Finds all unique elements and their positions
    unique, pos = np.unique(img_bylabel, return_inverse=True)

    # Count the number of each unique element
    counts = np.bincount(pos)

    # Select N most frequent labels
    seq_label = np.argpartition(counts, -num_canddt)[-num_canddt:]

    seq_comp_canddt = []
    for label in seq_label:  # Neglect the most frequent element, which is the background - i.e. 0
        # Make image (2D matrix) of zeros
        ret = np.zeros(shape=img_bylabel.shape, dtype=np.int32)

        # 1 if an element of img_bylabel matches with the label, remain 0 otherwise
        ret[img_bylabel == label] = 1
        plt.imshow(ret)
        plt.show()

        # Append only if the size exceeds the threshold
        # if np.sum(ret) > size_min: seq_comp_canddt.append(ret)
    return seq_comp_canddt


def make_seq_pair_distComp(seq_comp_canddt, img):
    # Calculate the centroid(1) of the entire image + some shift
    print(img.shape)
    centroid0_img = (img.shape[0] / 2 - 400, img.shape[1] * 1 / 2)
    centroid1_img = (img.shape[0] / 2 - 400, img.shape[1] * 1 / 2)

    # Sequence of mappings to return
    seq_pair_distComp = []
    for comp_canddt in seq_comp_canddt:
        # Calculate the centroid(2) of all non-zeros
        centroid_comp = ndimage.measurements.center_of_mass(comp_canddt)

        # Calculate Euclidean distance between the two centorids
        dist2center0 = ((centroid0_img[0] - centroid_comp[0]) ** 2 + (centroid0_img[1] - centroid_comp[1]) ** 2) ** 0.5
        dist2center1 = ((centroid1_img[0] - centroid_comp[0]) ** 2 + (centroid1_img[1] - centroid_comp[1]) ** 2) ** 0.5
        dist2center = min([dist2center0, dist2center1])

        # Map the distance to the component information, append it to the sequence to return
        pair_distComp = [dist2center, comp_canddt]
        seq_pair_distComp.append(pair_distComp)

    return seq_pair_distComp


def filter_sections(img_bylabel):
    # Sort by ascending order, so that the centermost elements be located at the foremost entries
    # seq_pair_distComp.sort()

    unique, counts = np.unique(img_bylabel, return_counts=True)
    print(zip(unique, counts))

    section0 = np.where(img_bylabel == 1, 1, 0)
    section1 = np.where(img_bylabel == 2, 1, 0)

    centroid_section0 = ndimage.measurements.center_of_mass(section0)
    centroid_img = (section0.shape[0]/2, section1.shape[1]/2)

    if centroid_section0[1] - centroid_img[1] > 0: return section1, section0
    else: return section0, section1

def get_bounds_vertical(section):
    for i in range(section.shape[0]):
        if np.sum(section[i]) > 0:
            bound_top = i
            break
    for i in range(section.shape[0]-1, 0, -1):
        if np.sum(section[i]) > 0:
            bound_bottom = i
            break
    return (bound_top, bound_bottom)

def make_clipper(bound, section, position, num_subsegment=3):
    height_target, width_target = section.shape
    height_segment = int((bound[1] - bound[0])/num_subsegment)
    modulo_term = (bound[1] - bound[0])%3
    if position == (num_subsegment-1): height_segment += modulo_term

    area_zeros_top = np.zeros(shape=(bound[0]+height_segment*position, width_target))
    area_clip = np.ones(shape=(height_segment, width_target))
    area_zeros_bottom = np.zeros(shape=(height_target - bound[0] - height_segment*(position+1), width_target))
    
    clipper = np.concatenate([area_zeros_top, area_clip, area_zeros_bottom], axis=0)
    return clipper

def make_seq_clipper(section, num_subsegment=3):
    bound = get_bounds_vertical(section)

    seq_clipper = []
    for position in range(num_subsegment): seq_clipper.append(make_clipper(bound, section, position))
    return seq_clipper

def make_seq_section_final(img_final, seq_clipper):
    seq_section_final = []
    for clipper in seq_clipper: seq_section_final.append(img_final*clipper)
    return seq_section_final

def reformat_fpath_dst(fpath_dst, i):
    fpath_tmp = fpath_dst.replace(".bmp", "_{}.bmp".format(i))
    return fpath_tmp

def save_result(section, fpath_dst):
    result = Image.fromarray(section.astype(np.uint8))
    result.save(fpath_dst)

def iterate_save_result(seq_section_final, fpath_dst):
    for i, section in enumerate(seq_section_final): save_result(section, reformat_fpath_dst(fpath_dst, i))

for fpath_src, fpath_dst in make_seq_fpath(FLAG_MODE, PATH_SRC, PATH_DST):
    try:
        print(fpath_dst)
        # Read data
        img = cv2.imread(fpath_src, 0)

        # # Step 1: Morphological Reconstruction
        # kernel = np.ones((3, 3), np.uint8)
        # erosion = cv2.erode(img, kernel, iterations=1)
        # opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        # dilation = cv2.dilate(opening, kernel, iterations=1)
        # closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        # img = closing

        # # Step 2: Calculate threshhold and convert into binary image
        # # Otsu's thresholding after Gaussian filtering
        # img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        # _, th_otsugauss = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)

        # # Rescale to {0, 1}
        # th_otsugauss = (th_otsugauss / 255 - 1) * (-1)

        # # Set the dtype into np.int32
        # kernel_ccl = np.asarray(th_otsugauss, dtype=np.int32)

        # Step 3: Connected Component Labelling

        # Label images by connected component
        kernel = np.where(img > 0, 1, 0)
        img_bylabel = measure.label(kernel)
        print(np.amax(img_bylabel))

        

        # plt.imshow(img_bylabel)
        # plt.show()

        # seq_comp_canddt = make_seq_comp_canddt(img_bylabel, 3, 10000)
        # print(len(seq_comp_canddt))

        # # Add distance information to each component
        # seq_pair_distComp = make_seq_pair_distComp(seq_comp_canddt, img)

        # Make section
        section0, section1 = filter_sections(img_bylabel)
        

        # section0, section1 = make_sections(seq_pair_distComp)
        # section = section0 + section1
        
        # img_final = section * img

        img_final0 = section0 * img
        img_final1 = section1 * img

        seq0_clipper = make_seq_clipper(section0)
        seq1_clipper = make_seq_clipper(section1)

        seq_section_final = make_seq_section_final(img_final0, seq0_clipper) + make_seq_section_final(img_final1, seq1_clipper)
        iterate_save_result(seq_section_final, fpath_dst)
        # save_result(img_final, fpath_dst)

        # Step 4: Give paddings
        # kernel_mr = np.ones((10,10), np.int32) # define structure for morphological reconstruction
        # section_mr = ndimage.binary_dilation(section, structure=kernel_mr, iterations=10) # dilation
        # section_mr = ndimage.binary_erosion(section_mr, structure=kernel_mr, iterations=5) # erosion
        # img_final = section_mr*img

        # Plots

            # plt.figure(figsize=(8, 8))

            # plt.subplot(331)
            # plt.imshow(kernel_ccl, cmap='gray')
            # plt.axis('off')

            # plt.subplot(332)
            # plt.imshow(closing, cmap='gray')
            # plt.axis('off')

            # plt.subplot(333)
            # plt.imshow(img_bylabel, cmap='nipy_spectral')
            # plt.axis('off')

            # for i, section in enumerate(seq_section_final):
            #     plt.subplot(334+i)
            #     plt.imshow(section, cmap='gray')
            #     plt.axis('off')

            # plt.tight_layout()
            # plt.show()

    except (IndexError, ValueError):
        print("Segmentation failed, move to the next file")