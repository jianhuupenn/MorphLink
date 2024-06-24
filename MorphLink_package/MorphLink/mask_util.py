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


#Remove overlapped channels
def remove_overlap(masks_index, dic_list, color1, color2, threshold=20):
	m_c=[]
	for i in range(len(masks_index)):
		index=masks_index[i]
		tmp=[]
		for j in range(len(index)):
			clusters=index[j]
			for k in clusters:
				tmp.append(dic_list[j][k])
		m_c.append(np.mean(np.array(tmp), 0).tolist())
	m_c=np.array(m_c)
	m1c=np.max(np.abs(m_c-np.array(color2)),1)
	m2c=np.max(np.abs(m_c-np.array(color2)),1)
	if (np.min(m1c) < threshold) and (np.min(m2c) < threshold):
		m1=np.argmin(m1c)
		m2=np.argmin(m2c)
		print("Mask 1 is ", m1, ", mask 2 is ", m2)
	else:
		print("Mask not found")
		return masks_index
	tmp=[]
	for i in range(len(masks_index[m2])):
		tmp.append([x for x in masks_index[m2][i] if x not in masks_index[m1][i]])
	masks_index[m2]=tmp
	return masks_index

def segment_patches(patches, save_dir="./seg_results",n_clusters=10, refine=True, refine_threshold=4):
	start_time = time.time()
	patch_size=patches.shape[1]
	for i in range(patches.shape[0]):
		print("Doing: ", i, "/", patches.shape[0])
		patch=patches[i]
		pixel_values = patch.reshape((-1, 3)) # reshape the image patch to have one row per pixel and 3 columns (RGB)
		#convert to float
		pixel_values = np.float32(pixel_values)
		seed=100
		random.seed(seed)
		np.random.seed(seed)
		kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixel_values)
		pred=kmeans.labels_
		#centroids=kmeans.cluster_centers_
		np.save(save_dir+"/patch"+str(i)+"_pred.npy", pred)
		if refine:
			#------------------------------------Refine------------------------------------------#
			refine_threshold=4 #least mode count
			pred_refined=refine_labels(pred=pred, patch_size=patch_size, refine_threshold=refine_threshold)
			np.save(save_dir+"/patch"+str(i)+"_pred_refined.npy", pred_refined)
	print("--- %s seconds ---" % (time.time() - start_time))

def get_color_dic(patches, seg_dir, save, save_dir, refine=True):
	patch_index=list(range(patches.shape[0]))
	patch_size=patches.shape[1]
	dic_list=[]
	for i in patch_index:
		if refine:
			pred_refined=np.load(seg_dir+"/patch"+str(i)+"_pred_refined.npy")
		else:
			pred_refined=np.load(seg_dir+"/patch"+str(i)+"_pred.npy")
		patch=patches[i].copy()
		c_m={} #cluster_marker expression
		c_a=[(j, np.sum(pred_refined==j)) for j in np.unique(pred_refined)] #cluster_area
		c_a=sorted(c_a, key=lambda x: x[1], reverse=True) # sorted in a descending order
		c_a=dict((x, y) for x, y in c_a)
		clusters=pred_refined.reshape(patch_size, patch_size)
		for j,  area_ratio in c_a.items():
			c_m[j]=np.median(patch[clusters==j], 0).tolist()
		dic_list.append(c_m)
		if save:
			with open(save_dir+"/patch"+str(i)+"_dic.pkl", 'wb') as f: pickle.dump(c_m, f)
	return dic_list

def combine_masks(masks, patch_info, img_size0, img_size1,center=False):
	#Combine masks to WSI
	patch_size=masks.shape[2]
	d0=int(np.ceil(img_size0/patch_size)*patch_size)
	d1=int(np.ceil(img_size1/patch_size)*patch_size)
	combined_masks=np.zeros([masks.shape[0], d0, d1])
	for i in range(masks.shape[0]): #Each mask
		print("Combining mask ", i)
		for j in range(masks.shape[1]): #Each patch
			info=patch_info.iloc[j]
			x, y=int(info["x"]), int(info["y"])
			if center:
				combined_masks[i, int(x-patch_size/2):int(x+patch_size/2), int(y-patch_size/2):int(y+patch_size/2)]=masks[i, j, :, :]
			else:
				combined_masks[i, int(x):int(x+patch_size), int(y):int(y+patch_size)]=masks[i, j, :, :]
	combined_masks=combined_masks[:, 0:img_size0, 0:img_size1]
	return combined_masks


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

def match_masks(dic_list, num_mask_each=10, mapping_threshold1=25, mapping_threshold2=50, min_no_mach_rate=0.2, min_unused_channel_rate=0.8):
	masks_index=[] #list of list of list, [mask[patch[cluster]]]
	used_channel=[]
	for i in range(len(dic_list)):
		print("Doing ", i)
		c_m0=dic_list[i]
		for c0 in list(c_m0.keys())[0:np.min([num_mask_each, len(c_m0)])]:
			if str(i)+"_"+str(c0) not in used_channel: #Start matching
				no_match=0
				masks_index_tmp=[]
				used_channel_tmp=[str(i)+"_"+str(c0)]
				for j in range(len(dic_list)):
					c_m1=dic_list[j]
					tmp1=[(k, np.max(np.abs(np.array(v)-np.array(c_m0[c0])))) for k, v in c_m1.items()]
					tmp2=list(filter(lambda x: x[1]<mapping_threshold1, tmp1))
					if len(tmp2)>0:
						masks_index_tmp.append([x[0] for x in tmp2])
						used_channel_tmp+=[str(j)+"_"+str(x[0])for x in tmp2]
						#print("Mapping: patch "+str(i)+" mask "+str(c0)+" to patch "+str(j)+" mask "+str(tmp[0][0])+"diff = "+str(np.round(tmp[0][1], 3)))
					else:
						tmp2=list(filter(lambda x: x[1]<mapping_threshold2, tmp1))
						if len(tmp2)>0:
							tmp2.sort(key=lambda x: x[1])
							masks_index_tmp.append([tmp2[0][0]])
							used_channel_tmp.append(str(j)+"_"+str(tmp2[0][0]))
						else:
							masks_index_tmp.append([])
							no_match+=1
				new_rate=1-len(set(used_channel_tmp)&set(used_channel))/len(used_channel_tmp)
				if no_match<(len(dic_list)*min_no_mach_rate) and (new_rate >= min_unused_channel_rate):
					print("Adding, ", "no match counts:",no_match, "new rate:",new_rate)
					masks_index.append(masks_index_tmp)
					used_channel+=used_channel_tmp
				else:
					print("Not adding, ", "no match counts:",no_match, "new rate:",new_rate)
	return masks_index

def calculate_patches_adj(patch_info, x_column="x", y_column="y"):
	adj=np.ones([patch_info.shape[0], patch_info.shape[0]])*-1
	for i in range(patch_info.shape[0]):
		for j in range(i, patch_info.shape[0]):
			point0, point1=patch_info.iloc[i][[x_column, y_column]].values,patch_info.iloc[j][[x_column, y_column]].values
			adj[i, j]=adj[j, i]=np.linalg.norm(point0 - point1)
	adj=pd.DataFrame(adj, index=list(range(patch_info.shape[0])), columns=list(range(patch_info.shape[0])))
	return adj


def extract_masks(masks_index, pred_file_locs, patch_size):
	num_masks=len(masks_index)
	masks=np.zeros([num_masks, len(masks_index[0]), patch_size, patch_size])
	for i in range(num_masks):
		print("Extracting mask ", i)
		mask_index=masks_index[i]
		for j in range(len(mask_index)):
			pred=np.load(pred_file_locs[j])
			mask = pred.reshape(patch_size, patch_size).astype(np.uint8)
			mask=1*np.isin(mask, mask_index[j])
			masks[i, j, :, :]=mask
	return masks

#Find masks in selected colors
def find_channels(color_list, m_c, m_a, threshold=20):
	ret=[]
	for color in color_list:
		diff=np.max(np.abs(np.array([color-v for k, v in m_c.items()])), 1)
		indexs=[i for i, x in enumerate(diff) if x<threshold]
		if len(indexs)>1:
			areas=[m_a[i] for i in indexs]
			ret.append(indexs[np.argmax(areas)])
		elif len(indexs)<1:
			print("Channel not found for", color, ", using the closest channel.")
			ret.append(np.argmin(diff))
		else:
			ret+=indexs
	return ret

"""
def match_masks(dic_list, num_mask_each=10, mapping_threshold=30, min_no_mach_rate=0.2, min_unused_channel_rate=0.9):
	masks_index=[]
	used_channel=[]
	for i in range(len(dic_list)):
		c_m0=dic_list[i]
		for c0 in list(c_m0.keys())[0:np.min([num_mask_each, len(c_m0)])]:
			if str(i)+"_"+str(c0) not in used_channel:
				used_channel.append(str(i)+"_"+str(c0))
				masks_index_tmp=[]
				used_channel_tmp=[]
				for j in range(len(dic_list)):
					c_m1=dic_list[j]
					tmp=[(k, np.max(np.abs(np.array(v)-np.array(c_m0[c0])))) for k, v in c_m1.items()]
					tmp.sort(key=lambda a: a[1])
					if tmp[0][1]<=mapping_threshold:
						masks_index_tmp.append(tmp[0][0])
						used_channel.append(str(j)+"_"+str(tmp[0][0]))
						used_channel_tmp.append(str(j)+"_"+str(tmp[0][0]))
						#print("Mapping: patch "+str(i)+" mask "+str(c0)+" to patch "+str(j)+" mask "+str(tmp[0][0])+"diff = "+str(np.round(tmp[0][1], 3)))
					else:
						masks_index_tmp.append(-1)
						print("No match", tmp[0][1])
				if masks_index_tmp.count(-1)<(len(masks_index_tmp)*min_no_mach_rate) and len(set(used_channel_tmp) & set(used_channel))>(len(masks_index_tmp)*min_unused_channel_rate):
					print("Adding, ", "-1 counts:",masks_index_tmp.count(-1), "new channels:",len(set(used_channel_tmp) & set(used_channel)))
					masks_index.append(masks_index_tmp)
				else:
					print("Not adding, ", "-1 counts:",masks_index_tmp.count(-1), "new channels:",len(set(used_channel_tmp) & set(used_channel)))
	masks_index=np.array(masks_index) #num_masks, #patches
	return masks_index



def match_masks_WSI(dic_list, adj, num_mask_each=10, mapping_threshold=30, min_no_mach_rate=0.2, min_unused_channel_rate=0.7):
	masks_index=[]
	used_channel=[]
	for i in range(len(dic_list)):
		c_m0=dic_list[i]
		for c0 in list(c_m0.keys())[0:np.min([num_mask_each, len(c_m0)])]:
			#Dict to record cluster_color for each mask mapping
			c_color={}
			if str(i)+"_"+str(c0) not in used_channel:
				used_channel.append(str(i)+"_"+str(c0))
				#Add color
				c_color[i]=c_m0[c0]
				masks_index_tmp=[]
				used_channel_tmp=[]
				for j in range(len(dic_list)):
					#Select nearest used patch
					nearest_patch_num=adj.loc[list(c_color.keys()),j].idxmin()
					print("Current patch:", j, "nearest patch:", nearest_patch_num)
					c_m1=dic_list[j]
					#tmp=[(k, np.max(np.abs(np.array(v)-np.array(c_m0[c0])))) for k, v in c_m1.items()]
					tmp=[(k, np.max(np.abs(np.array(v)-np.array(c_color[nearest_patch_num])))) for k, v in c_m1.items()]
					tmp.sort(key=lambda a: a[1])
					if tmp[0][1]<=mapping_threshold:
						masks_index_tmp.append(tmp[0][0])
						used_channel.append(str(j)+"_"+str(tmp[0][0]))
						used_channel_tmp.append(str(j)+"_"+str(tmp[0][0]))
						#Add color
						c_color[j]=c_m1[tmp[0][0]]
						#print("Mapping: patch "+str(i)+" mask "+str(c0)+" to patch "+str(j)+" mask "+str(tmp[0][0])+"diff = "+str(np.round(tmp[0][1], 3)))
					else:
						masks_index_tmp.append(-1)
						print("No match", tmp[0][1])
				if masks_index_tmp.count(-1)<(len(masks_index_tmp)*min_no_mach_rate) and len(set(used_channel_tmp) & set(used_channel))>(len(masks_index_tmp)*min_unused_channel_rate):
					print("Adding, ", "-1 counts:",masks_index_tmp.count(-1), "new channels:",len(set(used_channel_tmp) & set(used_channel)))
					masks_index.append(masks_index_tmp)
				else:
					print("Not adding, ", "-1 counts:",masks_index_tmp.count(-1), "new channels:",len(set(used_channel_tmp) & set(used_channel)))
	masks_index=np.array(masks_index) #num_masks, #patches
	return masks_index

"""
