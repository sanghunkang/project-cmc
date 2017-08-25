#!/usr/bin/python
# -*- coding: utf-8 -*-
#############################################################################
# Import built-in modules
import os
import pickle as pickle

# Import 3rd party packages
import PIL.Image as Image
import numpy as np


# Define constants (mostly paths of directories)
DIR_DATA_LABELLED_C00_TRAIN = "C:\\dev-data\\labelled-C00-train\\"
DIR_DATA_LABELLED_CNN_TRAIN = "C:\\dev-data\\labelled-CNN-train\\"
DIR_DATA_LABELLED_C00_TEST = "C:\\dev-data\\labelled-C00-test\\"
DIR_DATA_LABELLED_CNN_TEST = "C:\\dev-data\\labelled-CNN-test\\"
DIR_DATA_PICKLE = "C:\\dev-data\\pickle\\"

DIR_DATA_LABELLED_C0N = "C:\\dev-data\\labelled-C0N"
DIR_DATA_LABELLED_C1N = "C:\\dev-data\\labelled-C1N"
DIR_DATA_LABELLED_C2N = "C:\\dev-data\\labelled-C2N"
DIR_DATA_LABELLED_C3N = "C:\\dev-data\\labelled-C3N"

resolution = (448, 448, 3)
n_class = 2 # Normal & abnormal

def dir_to_pickle(dir_src, resolution, vec_class):
	len_h, len_w, len_c = resolution

	seq_fpath = os.listdir(dir_src)

	seq_rec = np.zeros(shape=(len(seq_fpath), len_h*len_w*len_c + n_class), dtype=np.float32)
	for i, fpath in enumerate(seq_fpath):
		print(fpath)
		img = Image.open(os.path.join(dir_src, fpath)).convert('RGB')
		# img = raw_img.convert('RGB')
		size_img = img.size
		
		min_side = min(size_img)
		padding_h, padding_v= (size_img[0] - min_side)/2, (size_img[1] - min_side)/2
		
		img_crop = img.crop((padding_h, padding_v, size_img[0] - padding_h, size_img[1] - padding_v))
		# img_crop = img_crop.crop((img_crop.size[0]*0.15, img_crop.size[1]*0, img_crop.size[0]*0.85, img_crop.size[1]*0.70))
		# img_crop.show()


		img_resize = img_crop.resize((len_h, len_w))

		arr_img = np.asarray(img_resize)
		arr1d_img = arr_img.reshape(len_h*len_w*len_c)
		seq_rec[i] = np.append(arr1d_img, vec_class) # [1, 0] for normal

	return seq_rec

def divide_trainNtest(seq_rec, ratio_train):
	np.random.shuffle(seq_rec)
	seq_rec_train = seq_rec[:int(seq_rec.shape[0]*ratio_train)]
	seq_rec_test = seq_rec[int(seq_rec.shape[0]*ratio_train):]
	return seq_rec_train, seq_rec_test

# seq_rec_C00_train = dir_to_pickle(DIR_DATA_LABELLED_C00_TRAIN, resolution, [1, 0])
# seq_rec_C00_test = dir_to_pickle(DIR_DATA_LABELLED_C00_TEST, resolution, [1, 0])
#
# seq_rec_C00 = np.concatenate([seq_rec_C00_test, seq_rec_C00_train])
seq_rec_C0N = dir_to_pickle(DIR_DATA_LABELLED_C0N, resolution, [1, 0])
seq_rec_C1N = dir_to_pickle(DIR_DATA_LABELLED_C1N, resolution, [1, 0])
seq_rec_C2N = dir_to_pickle(DIR_DATA_LABELLED_C2N, resolution, [0, 1])
seq_rec_C3N = dir_to_pickle(DIR_DATA_LABELLED_C3N, resolution, [0, 1])

# print(seq_rec_C00.shape)
print(seq_rec_C0N.shape)
print(seq_rec_C1N.shape)
print(seq_rec_C2N.shape)
print(seq_rec_C3N.shape)

seq_rec_ltet10 = seq_rec_C1N
seq_rec_gt10 = seq_rec_C2N

np.random.shuffle(seq_rec_ltet10)
np.random.shuffle(seq_rec_gt10)

seq_rec_ltet10 = seq_rec_ltet10[:450]
seq_rec_gt10 = seq_rec_gt10[:450]

print(seq_rec_ltet10.shape)
print(seq_rec_gt10.shape)

seq_rec_train_ltet10, seq_rec_test_ltet10 = divide_trainNtest(seq_rec_ltet10, 0.8)
seq_rec_train_gt10, seq_rec_test_gt10 = divide_trainNtest(seq_rec_ltet10, 0.8)

seq_rec_train = np.concatenate([seq_rec_train_ltet10, seq_rec_test_ltet10])
seq_rec_test = np.concatenate([seq_rec_train_gt10, seq_rec_test_gt10])

# For compatibility reason, the highest protocol is not used.
with open(DIR_DATA_PICKLE + "data_train_th20_100.pickle", 'wb') as handle:
	pickle.dump(seq_rec_train, handle, protocol=2)

with open(DIR_DATA_PICKLE + "data_test_th20_100.pickle", 'wb') as handle:
	pickle.dump(seq_rec_test, handle, protocol=2)
