#Packed main functions
import os,csv,re, time, pickle
import random
import pandas as pd
import numpy as np
import torch
from scipy import stats
import scanpy as sc
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2
from skimage import io
from skimage.util import img_as_ubyte
import slideio
from sklearn.cluster import KMeans
import pickle
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import argparse
import imutils

#Local import
from .util import *
from .contour_util import *
from .calculate_dis import *
from .extract_hancraft_features_util import *
from .tissue_patch_util import patch_split, patch_split_for_ST, patch_split_with_mask, combine_patches
from .convolve2D import convolve2D_clean, convolve2D_refine, convolve2D_nbr_num
from .refine_and_merge_util import refine_labels, merge_labels
from .mask_util import match_masks, calculate_patches_adj, extract_masks, segment_patches, get_color_dic, combine_masks, remove_overlap, find_channels
from .cc_util import extract_cc, plot_cc, combine_CC_patches

def step1_Preprocessing(plot_dir, slide, block=None, cnt_threshold=5000):
	if not os.path.exists(plot_dir):
		os.mkdir(plot_dir)
		os.mkdir(plot_dir+"/results")
		os.mkdir(plot_dir+"/seg_results")
	scene = slide.get_scene(0)
	if block!=None:
		img = scene.read_block(block) #lr, ud
	else:
		img=scene.read_block()
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	#Contour detection
	cnts=cv2_detect_contour(img, all_cnt_info=True)
	tissue_mask = np.zeros(img.shape[0:2])
	for c in cnts:
		if c[2]>cnt_threshold:
			_=cv2.drawContours(tissue_mask, [c[0]], -1, (1), thickness=-1)
	return img, tissue_mask

def step2_Get_Patches(plot_dir, img, tissue_mask, patch_size=2000, tissue_threshold=0.05):
	patch_info, patches, img_extended, tissue_mask_extended =patch_split_with_mask(img, patch_size, tissue_mask, tissue_threshold) #patch=[x:x+size, y:y+size]
	patch_info.to_csv(plot_dir+"/results/patch_info.csv")
	np.save(plot_dir+"/results/patches.npy", patches)
	cv2.imwrite(plot_dir+'/img_extended.png', img_extended)
	d0, d1=img_extended.shape[0],img_extended.shape[1] 
	return d0, d1


def step3_Stain_Normalization(plot_dir, d0, d1, template_path, template_size, color_list=[[255, 255, 255], [0, 0,255], [255, 0, 0]]):
	patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
	patches=np.load(plot_dir+"/results/patches.npy")
	template=cv2.imread(template_path)
	template_pool, template_vec=build_template_pool(img=template, patch_size=template_size, color_list=color_list)
	patches_norm=patches.copy()
	for i in range(patch_info.shape[0]):
		patches_norm[i]=stain_normalization_muti_temp(patches[i], color_list, template_pool, template_vec)
	img_extended_norm=combine_patches(patches_norm, patch_info, d0, d1,center=False)
	np.save(plot_dir+"/results/patches_norm.npy", patches_norm)
	cv2.imwrite(plot_dir+'/img_extended_norm.png', img_extended_norm)

def step4_Segmentation(plot_dir, n_clusters=10, refine=True, refine_threshold=4):
	save_dir=plot_dir+"/seg_results"
	patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
	patches=np.load(plot_dir+"/results/patches.npy")
	segment_patches(patches, save_dir=save_dir,n_clusters=n_clusters, refine=refine, refine_threshold=refine_threshold)
	dic_list=get_color_dic(patches, seg_dir=save_dir, save=False, save_dir=save_dir, refine=True)
	with open(save_dir+'/dic_list.pickle.pkl', 'wb') as f:
		pickle.dump(dic_list, f)

def check_dic_list(plot_dir):
	save_dir=plot_dir+"/seg_results"
	if not os.path.exists(save_dir+'/dic_list.pickle.pkl'):
		patches=np.load(plot_dir+"/results/patches.npy")
		dic_list=get_color_dic(patches, seg_dir=save_dir, save=False, save_dir=save_dir, refine=True)
		with open(save_dir+'/dic_list.pickle.pkl', 'wb') as f:
			pickle.dump(dic_list, f)

def step5_Extract_Masks(plot_dir, patch_size, num_mask_each, mapping_threshold1, mapping_threshold2, min_no_mach_rate=1, min_unused_channel_rate=0.5):
	save_dir=plot_dir+"/seg_results"
	patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
	with open(save_dir+'/dic_list.pickle.pkl', 'rb') as f:
		dic_list = pickle.load(f)
	masks_index=match_masks(dic_list, num_mask_each=num_mask_each, mapping_threshold1=mapping_threshold1, mapping_threshold2=mapping_threshold2,min_no_mach_rate=min_no_mach_rate, min_unused_channel_rate=min_unused_channel_rate)
	pred_file_locs=[save_dir+"/patch"+str(j)+"_pred_refined.npy" for j in range(patch_info.shape[0])]
	masks=extract_masks(masks_index=masks_index, pred_file_locs=pred_file_locs[0:patch_info.shape[0]], patch_size=patch_size)
	np.save(plot_dir+"/results/masks_"+str(mapping_threshold1)+"_"+str(mapping_threshold2)+".npy", masks)
	with open(plot_dir+'/results/masks_index_'+str(mapping_threshold1)+"_"+str(mapping_threshold2)+'.pkl', 'wb') as f:
		pickle.dump(masks_index, f)
	return masks, masks_index

def step5_Extract_Masks_multi_sections(plot_dirs, patch_sizes, num_mask_each, mapping_threshold1, mapping_threshold2, min_no_mach_rate=1, min_unused_channel_rate=0.5):
	dic_lists=[]
	for i in range(len(plot_dirs)):
		plot_dir=plot_dirs[i]
		save_dir=plot_dir+"/seg_results"
		with open(save_dir+'/dic_list.pickle.pkl', 'rb') as f:
			dic_list = pickle.load(f)
		dic_lists+=dic_list
	masks_indexs=match_masks(dic_lists, num_mask_each=num_mask_each, mapping_threshold1=mapping_threshold1, mapping_threshold2=mapping_threshold2,min_no_mach_rate=min_no_mach_rate, min_unused_channel_rate=min_unused_channel_rate)
	counter=0
	for i in range(len(plot_dirs)):
		plot_dir=plot_dirs[i]
		patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
		patch_size=patch_sizes[i]
		save_dir=plot_dir+"/seg_results"
		pred_file_locs=[save_dir+"/patch"+str(j)+"_pred_refined.npy" for j in range(patch_info.shape[0])]
		masks_index=[]
		for j in range(len(masks_indexs)):
			masks_index.append(masks_indexs[j][counter:counter+patch_info.shape[0]])
		counter+=patch_info.shape[0]
		masks=extract_masks(masks_index=masks_index, pred_file_locs=pred_file_locs[0:patch_info.shape[0]], patch_size=patch_size)
		np.save(plot_dir+"/results/masks_"+str(mapping_threshold1)+"_"+str(mapping_threshold2)+".npy", masks)
		with open(plot_dir+'/results/masks_index_'+str(mapping_threshold1)+"_"+str(mapping_threshold2)+'.pkl', 'wb') as f:
			pickle.dump(masks_index, f)

def step6_Plot_Masks(plot_dir, d0, d1, masks, patch_size, mapping_threshold1, mapping_threshold2):
	plot_mask_dir=plot_dir+"/masks_"+str(mapping_threshold1)+"_"+str(mapping_threshold2)
	patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
	if not os.path.exists(plot_mask_dir):
		os.mkdir(plot_mask_dir)
	combined_masks=combine_masks(masks, patch_info, d0, d1,center=False)
	for i in range(masks.shape[0]): #Each mask
		print("Plotting mask ", str(i))
		ret=(combined_masks[i]*255)
		cv2.imwrite(plot_mask_dir+'/mask'+str(i)+'.png', ret.astype(np.uint8))

def step7_Select_Masks(plot_dir, d0, d1):
	color1=np.array([28, 14, 77]) #Nuclei
	color2=np.array([20,  12, 132])#Myocyte
	color3=np.array([154, 109, 119]) #Stroma
	img_extended_norm=cv2.imread(plot_dir+'/img_extended_norm.png')
	patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
	masks1=np.load(plot_dir+"/results/masks_10_40.npy")
	with open(plot_dir+'/results/masks_index_10_40.pkl', 'rb') as f:
		masks_index1 = pickle.load(f)
	masks2=np.load(plot_dir+"/results/masks_50_60.npy")
	with open(plot_dir+'/results/masks_index_50_60.pkl', 'rb') as f:
		masks_index2 = pickle.load(f)
	#-----------------------------------Find channels------------------------------------------#
	combined_masks1=combine_masks(masks1, patch_info, d0, d1,center=False)
	m_c1={}
	m_a1={}
	for i in range(combined_masks1.shape[0]):
		mask=combined_masks1[i]
		m_c1[i]=np.median(img_extended_norm[mask>0], 0)
		m_a1[i]=np.sum(mask)
	combined_masks2=combine_masks(masks2, patch_info, d0, d1,center=False)
	m_c2={}
	m_a2={}
	for i in range(combined_masks2.shape[0]):
		mask=combined_masks2[i]
		m_c2[i]=np.median(img_extended_norm[mask>0], 0)
		m_a2[i]=np.sum(mask)
	channel_list1=find_channels(color_list=[color1, color2, color3], m_c=m_c1, m_a=m_a1, threshold=30)
	channel_list2=find_channels(color_list=[color1, color2, color3], m_c=m_c2, m_a=m_a2, threshold=30)
	channel_list3=find_channels(color_list=[color1, color2, color3], m_c=m_c2, m_a=m_a2, threshold=30)
	masks_selected=np.stack((masks1[channel_list1[0]], masks2[channel_list2[1]], masks2[channel_list3[2]]))
	masks_index_selected=[masks_index1[channel_list1[0]], masks_index2[channel_list2[1]], masks_index2[channel_list2[2]]]
	combined_masks=combine_masks(masks_selected, patch_info, d0, d1,center=False)
	#----Plot----
	for i in range(combined_masks.shape[0]): #Each mask
		ret=(combined_masks[i]*255)
		cv2.imwrite(plot_dir+'/selected_mask'+str(i)+'.png', ret.astype(np.uint8))
	#----Save----
	np.save(plot_dir+"/results/masks_selected.npy", masks_selected)
	with open(plot_dir+'/results/masks_index_selected.pkl', 'wb') as f:
		pickle.dump(masks_index_selected, f)

def step8_CC_Detection_for_ST(plot_dir, patch_info, masks_selected, masks_index_selected, save_dir=None, details=False, center=False,):
	if save_dir==None:
		save_dir=plot_dir+"/results"
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	patch_size=masks_selected.shape[2]
	pred_file_locs=[plot_dir+"/seg_results/patch"+str(j)+"_pred_refined.npy" for j in range(patch_info.shape[0])]
	#No detail, for nuclei and body
	label_area_list, labels=extract_cc(masks_index_selected, pred_file_locs, patch_size, connectivity=4, min_area=25, details=False)
	np.save(save_dir+"/cc_no_details.npy", labels)
	if details:
		#Details, body only
		label_area_list, labels=extract_cc(masks_index_selected, pred_file_locs, patch_size, connectivity=4, min_area=25, details=True)
		np.save(save_dir+"/cc_details.npy", labels)

def step8_CC_Detection(plot_dir, save_dir=None, details=True):
	if save_dir==None:
		save_dir=plot_dir+"/results"
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)
	patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
	masks_selected=np.load(plot_dir+"/results/masks_selected.npy")
	with open(plot_dir+'/results/masks_index_selected.pkl', 'rb') as f:
		masks_index_selected = pickle.load(f)
	patch_size=masks_selected.shape[2]
	pred_file_locs=[plot_dir+"/seg_results/patch"+str(j)+"_pred_refined.npy" for j in range(patch_info.shape[0])]
	#-----------------------------------CC Detection------------------------------------------#
	#No detail, for nuclei and body
	label_area_list, labels=extract_cc(masks_index_selected[0:2], pred_file_locs, patch_size, connectivity=4, min_area=25, details=False)
	np.save(save_dir+"/cc_no_details.npy", labels)
	if details:
		#Details, body only
		label_area_list, labels=extract_cc([masks_index_selected[1]], pred_file_locs, patch_size, connectivity=4, min_area=25, details=True)
		np.save(save_dir+"/cc_details.npy", labels)

def step9_Filter_Nuclei(plot_dir, d0, d1, channel=0, min_area=10, max_area=1000, min_solidity=0.4, max_hw_ratios=6):
	patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
	labels=np.load(plot_dir+"/results/cc_no_details.npy")
	ret=combine_CC_patches(labels=labels, patch_info=patch_info, channel=channel, d0=d0, d1=d1)
	cc_features=Extract_CC_Features_each_CC(labels=ret,  GLCM=False, img=None)
	cc_features["hw_ratios"]=cc_features["major_axis_length"]/cc_features["minor_axis_length"]
	filtered_cc_index=cc_features[(cc_features["solidity"]>min_solidity) 
								& (cc_features["hw_ratios"]<max_hw_ratios)
								& (cc_features["area"]>min_area) 
								& (cc_features["area"]<max_area)]["label"].tolist()
	np.save(plot_dir+"/results/nuclei_cc.npy", ret)
	tmp=ret*(np.isin(ret, filtered_cc_index))
	np.save(plot_dir+"/results/nuclei_cc_filtered.npy", tmp)


def step10_Filter_Body_IMUS(plot_dir, d0, d1, channel=1, min_area=2000, max_area=10000):
	patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
	#-------------- filter cell body on NO_details
	labels=np.load(plot_dir+"/results/cc_no_details.npy")
	ret=combine_CC_patches(labels=labels, patch_info=patch_info, channel=channel, d0=d0, d1=d1)
	region_props = regionprops(ret.astype(int))
	label_area=[(prop.label, prop.area) for prop in region_props]
	l=list(filter(lambda x: x[1]>min_area and x[1]<max_area, label_area))
	l=[int(x[0]) for x in l]
	ret = ret*np.isin(ret, l)
	np.save(plot_dir+"/results/cell_body_cc_filtered_no_details.npy", ret)
	#--------------------filter cell body on details using neclei------------------------------
	nuclei_cc=np.load(plot_dir+"/results/nuclei_cc_filtered.npy")
	labels=np.load(plot_dir+"/results/cc_details.npy")
	cell_body_cc=combine_CC_patches(labels=labels, patch_info=patch_info, channel=0, d0=d0, d1=d1)
	ret=np.zeros(cell_body_cc.shape[0:2])
	counter=1
	num_cc=len(np.unique(nuclei_cc))
	current_num=0
	for i in (np.unique(nuclei_cc).tolist()):
		if current_num%100==0:
			print("Doing ", str(current_num), "/", str(num_cc))
		current_num+=1
		if i>0:
			cc=np.array(nuclei_cc==i, dtype=np.uint8)
			# Find contours in img.
			cnt=cv2_detect_contour(cc, apertureSize=5,L2gradient = True)
			# Find the contour with the maximum area.
			cnt=scale_contour(cnt, 2)
			cc=cv2.drawContours(cc, [cnt], contourIdx=-1, color=(1), thickness=-1)
			#cv2.imwrite('ttt.png', (cc*255).astype(np.uint8))
			cc_indexs=np.unique(cell_body_cc[cc!=0])
			cc_indexs=cc_indexs[cc_indexs != 0]
			if len(cc_indexs)>1:
				#Do filter here
				tmp=(np.isin(cell_body_cc, cc_indexs)*1).astype(np.uint8)
				cnts = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  
				# Find the contour with the maximum area.
				c = max(cnts, key=cv2.contourArea)
				rect = cv2.minAreaRect(c)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				w, h=np.linalg.norm(box[0] - box[1])+1, np.linalg.norm(box[0] - box[3])+1
				w, h=np.min([w, h]), np.max([w, h])
				cc_area=np.sum(tmp)
				box_area=w*h
				area_ratio=cc_area/box_area
				hw_ratio=h/w
				if (cc_area>min_area) and (cc_area<max_area) and (area_ratio>0.4) and (hw_ratio<8):
					ret[np.isin(cell_body_cc, cc_indexs)]=counter
					counter+=1
	np.save(plot_dir+"/results/cell_body_cc_filtered_details.npy", ret)
	#------------------------------Combine cell body on No details and details------------------------------
	cc1=np.load(plot_dir+"/results/cell_body_cc_filtered_no_details.npy")
	cc2=np.load(plot_dir+"/results/cell_body_cc_filtered_details.npy")
	cc2[cc2!=0]=cc2[cc2!=0]+np.max(cc1)
	ret=cc1.copy()
	ret[ret==0]=cc2[ret==0]
	np.save(plot_dir+"/results/cell_body_cc_combined.npy", ret)




#----------------------------Watershed----------------------#
def step11_Watershed(plot_dir, d0, d1, channel_nuclei=0,channel_body=1):
	patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
	masks_selected=np.load(plot_dir+"/results/masks_selected.npy")
	combined_masks=combine_masks(masks_selected, patch_info, d0, d1,center=False)
	body=combined_masks[channel_body]
	nuclei=combined_masks[channel_nuclei]
	#Fill body with nuclei
	body[nuclei!=0]=1
	thresh=body.astype(np.uint8)
	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)
	# perform a connected component analysis on the local peaks,
	# using 8-connectivity, then appy the Watershed algorithm
	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-D, markers, mask=thresh)
	np.save(plot_dir+"/results/cc_watershed.npy", labels)

def step12_Refine(plot_dir, labels_path, kernel1, kernel2, kernel3, hole_filling_threshold=50):
	cc=np.load(labels_path)
	ret=np.zeros(cc.shape)
	ccs=np.unique(cc).tolist()
	ccs.remove(0)
	for c in ccs:
		tmp=((cc==c)*1).astype(np.uint8)
		if np.sum(tmp)>1000:
			tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel1)
			tmp = cv2.dilate(tmp, kernel2, iterations=1)
			cnt_info=cv2_detect_contour(tmp, all_cnt_info=True)
			for i in range(len(cnt_info)):
				if cnt_info[i][2]<hole_filling_threshold:
					cnt=cnt_info[i][0]
					_=cv2.drawContours(tmp, [cnt], -1, (1), thickness=-1)
			tmp=cv2.erode(tmp, kernel3, iterations=1)
			ret[tmp!=0]=c
	np.save(labels_path.rstrip(".npy")+"_processed.npy", ret)

#----------------------------Feature extraction----------------------#
#labels_path=plot_dir+"/results/watershed_processed.npy"
#labels_path=plot_dir+"/results/cell_body_cc_combined.npy"

def step13_Extract_Handcraft_Features(plot_dir, d0, d1, labels_path, out_file_name):
	labels=np.load(labels_path)
	img_extended_norm=cv2.imread(plot_dir+'/img_extended_norm.png')
	features= Extract_CC_Features_each_CC(labels=labels, GLCM=True, img=img_extended_norm)
	features["major_minor_axis_ratio"]=features["major_axis_length"]/features["minor_axis_length"]
	features["form_factor"]=4*np.pi*features["area"]/(features["perimeter"]**2)
	features.to_csv(plot_dir+"/results/"+out_file_name)

#Filter CC based on features from step13
def step14_Filter_Labels(plot_dir, d0, d1, 
						labels_path, 
						in_file_name,
						min_minor_axis_length, 
						max_minor_axis_length, 
						min_major_minor_axis_ratio, 
						min_solidity, 
						min_area, 
						max_area, 
						min_form_factor):
	labels=np.load(labels_path)
	features=pd.read_csv(plot_dir+"/results/"+in_file_name, header=0, index_col=0)
	ids=features[(features["minor_axis_length"]>min_minor_axis_length) 
		& (features["minor_axis_length"]<max_minor_axis_length)
		& (features["major_minor_axis_ratio"]>min_major_minor_axis_ratio)
		& (features["solidity"]>min_solidity)
		&(features["area"]>min_area) 
		& (features["area"]<max_area)
		& (features["form_factor"]>min_form_factor)]["label"].tolist()
	tmp=(np.isin(labels, ids)*1).astype(np.uint8)
	labels=labels*tmp
	np.save(labels_path.rstrip(".npy")+"_filtered.npy", labels)

#labels_path=plot_dir+"/results/watershed_processed.npy"
#labels_path=plot_dir+"/results/cell_body_cc_combined.npy"
#labels_path=plot_dir+"/results/cell_body_cc_combined_filtered.npy"

def step15_Extract_Radial_Intensity_features(plot_dir, d0, d1, labels_path, in_file_name, out_file_name):
	labels=np.load(labels_path)
	img_extended_norm=cv2.imread(plot_dir+'/img_extended_norm.png')
	features=pd.read_csv(plot_dir+"/results/"+in_file_name, header=0, index_col=0)
	ret=radial_profile_features(labels, img_extended_norm, features, centers=None)
	ret.to_csv(plot_dir+"/results/"+out_file_name)


