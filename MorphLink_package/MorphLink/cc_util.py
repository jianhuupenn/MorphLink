import os,csv,re, time, pickle
import random
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
#Local import
from .convolve2D import convolve2D_clean, convolve2D_refine, convolve2D_nbr_num
from .refine_and_merge_util import refine_labels, merge_labels


def Connected_Components_Separation(mask, connectivity=4, min_area=20):
	num_cc, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
	#Sort by area
	label_area=[]
	for i in range(num_cc):
		#Filter out background
		if np.median(mask[labels==i])!=0:
			label_area.append((i, np.sum(labels==i)))
	label_area=filter(lambda x: (x[1] > min_area), label_area)
	label_area=sorted(label_area, key=lambda x: x[1], reverse=True)
	labels=labels*np.isin(labels, [x[0] for x in label_area])
	return label_area, labels


def extract_cc(masks_index, pred_file_locs, patch_size, connectivity=4, min_area=25, details=True):
	num_masks=len(masks_index)
	ret1=[] #label_area
	ret2=np.zeros([num_masks, len(masks_index[0]), patch_size, patch_size]) #labels
	for i in range(num_masks):
		print("Extracting CC for mask ", i)
		mask_index=masks_index[i]
		#init
		label_area_mask=[]
		for j in range(len(mask_index)):
			labels=np.zeros([patch_size, patch_size])
			index=mask_index[j]
			pred=np.load(pred_file_locs[j])
			pred = pred.reshape(patch_size, patch_size).astype(np.uint8)
			#init
			if details:
				labels=np.zeros([patch_size, patch_size])
				label_area_mask_patch=[]
				base=0
				for k in index:
					pred_tmp=1*(pred==k)
					#mask_tmp=convolve2D_clean(mask_tmp, kernel, padding=1, strides=1, padding_value=0, threshold=6)
					label_area_tmp, labels_tmp=Connected_Components_Separation(pred_tmp, connectivity=connectivity, min_area=min_area)
					label_area_tmp=[(x[0]+base, x[1]) for x in label_area_tmp]
					labels_tmp=labels_tmp+(labels_tmp!=0)*base
					base=max(label_area_tmp, key = lambda i : i[0])[0]
					label_area_mask_patch+=label_area_tmp
					labels+=labels_tmp
			else:
				pred_tmp=np.isin(pred, index)
				label_area_tmp, labels_tmp=Connected_Components_Separation(pred_tmp, connectivity=connectivity, min_area=min_area)
				label_area_mask_patch=label_area_tmp
				labels=labels_tmp
			label_area_mask.append(label_area_mask_patch)
			labels=labels.astype("int")
			ret2[i, j, :, :]=labels
		ret1.append(label_area_mask)
	#ret2 mask x patch x size x size
	ret2=ret2.astype("int")
	return ret1, ret2


def plot_cc(labels):
	ret=np.zeros([labels.shape[0], labels.shape[1], 3])
	for i in np.unique(labels):
		if i!=0:
			color=[random.randint(0,255),random.randint(0,255), random.randint(0,255)]
			ret[labels==i]=color
	ret=ret.astype(np.uint8)
	return ret


def combine_CC_patches(labels, patch_info, channel, d0, d1, center=False):
	#CC index numbers in different patches may overlap! Use 'base' to reindex
	patch_size=labels.shape[2]
	ret=np.zeros([d0, d1])
	base=0
	for j in range(labels.shape[1]):
		info=patch_info.iloc[j]
		x, y=int(info["x"]), int(info["y"])
		if center:
			ret[int(x-patch_size/2):int(x+patch_size/2), int(y-patch_size/2):int(y+patch_size/2)]=labels[channel, j, :, :]+(labels[channel, j, :, :]!=0)*base
		else:
			ret[int(x):int(x+patch_size), int(y):int(y+patch_size)]=labels[channel, j, :, :]+(labels[channel, j, :, :]!=0)*base
		base=np.max(ret)
	return ret





