#Automatical template generation and selection
import os,csv,re, time, pickle
import pandas as pd
import numpy as np
import torch
from scipy import stats
import scanpy as sc
import staintools
from .tissue_patch_util import patch_split, patch_split_for_ST, patch_split_with_mask



def simple_seg(sample, color_list):
	diff=np.zeros([sample.shape[0], sample.shape[1], len(color_list)])
	for i in range(len(color_list)):
		color=color_list[i]
		tmp=np.linalg.norm(sample-np.array(color), axis=2)
		diff[..., i]=tmp
	pred=np.argmin(diff, axis=2)
	return pred


def find_dominate_color(img, resize=True, merge_threshold=20):
	if resize:
		resize_factor=500/np.min(img.shape[0:2])
		resize_width=int(img.shape[1]*resize_factor)
		resize_height=int(img.shape[0]*resize_factor)
		img = cv2.resize(img, (resize_width, resize_height), interpolation = cv2.INTER_AREA)
	pixel_values = img.reshape((-1, 3))
	#convert to float
	pixel_values = np.float32(pixel_values)
	kmeans = KMeans(n_clusters=10, random_state=0).fit(pixel_values)
	pred=kmeans.labels_
	color_list=kmeans.cluster_centers_.astype("int")
	return color_list


def build_template_pool(img, patch_size, color_list, tissue_threshold=0.05):
	tissue_threshold=0.05
	tissue_mask=np.ones([img.shape[0], img.shape[1]])
	patch_info, patches, _, _ =patch_split_with_mask(img, patch_size, tissue_mask, tissue_threshold) #patch=[x:x+size, y:y+size]
	template_vec=np.zeros([patches.shape[0], len(color_list)])
	for i in range(patches.shape[0]):
		pred=simple_seg(patches[i], color_list)
		for j in range(len(color_list)):
			template_vec[i, j]=np.sum(pred==j)/(pred.shape[0]*pred.shape[1])
	return patches, template_vec

def find_template(sample, color_list, template_vec):
	num_color_used=len(color_list)-1
	pred=simple_seg(sample,color_list)
	sample_vec=[np.sum(pred==j)/(pred.shape[0]*pred.shape[1]) for j in range(len(color_list))]
	sample_vec=np.array(sample_vec)
	template_index=np.argmin(np.sum(np.square(template_vec[:, 0:num_color_used]-sample_vec[:, 0:num_color_used]), 1))
	return template_index

def stain_normalization_muti_temp(sample, color_list, template_pool, template_vec, luminosity=True, method="macenko"):
	if luminosity:
		sample = staintools.LuminosityStandardizer.standardize(sample)
	pred=simple_seg(sample, color_list)
	label_colours=np.array(color_list)
	pred_rgb = np.array([label_colours[ c % label_colours.shape[0] ] for c in pred])
	sample_vec=[np.sum(pred==j)/(pred.shape[0]*pred.shape[1]) for j in range(len(color_list))]
	sample_vec=np.array(sample_vec)
	template_index=np.argmin(np.sum(np.square(template_vec-sample_vec), 1))
	normalizer = staintools.StainNormalizer(method=method)
	normalizer.fit(template_pool[template_index])
	sample = normalizer.transform(sample)
	return sample




