import pandas as pd
import numpy as np
import cv2

def patch_split(img, patch_size, contours_tissue, holes_tissue, tissue_threshold=0.2):
	#=====================================Create mask=====================================================
	tissue_mask=np.zeros(img.shape[0:2])
	for cnt in contours_tissue:
		cv2.drawContours(tissue_mask, [cnt], -1, (1), thickness=-1)
	for holes in holes_tissue:
		for cnt in holes:
			cv2.drawContours(tissue_mask, [cnt], -1, (0), thickness=-1)
	return patch_split_with_mask(img, patch_size, tissue_mask, tissue_threshold)


def patch_split_with_mask(img, patch_size, tissue_mask, tissue_threshold=0.2):
	#=====================================Extend image boundary using black===============================
	d0=int(np.ceil(img.shape[0]/patch_size)*patch_size)
	d1=int(np.ceil(img.shape[1]/patch_size)*patch_size)
	#bg_color=np.median(img[tissue_mask==0], axis=0)
	img_extended=np.concatenate((img, np.zeros((img.shape[0], d1-img.shape[1], 3), dtype=np.uint8)), axis=1)
	img_extended=np.concatenate((img_extended, np.zeros((d0-img.shape[0], d1, 3), dtype=np.uint8)), axis=0)
	tissue_mask_extended=np.concatenate((tissue_mask, np.zeros((img.shape[0], d1-img.shape[1]), dtype=np.uint8)), axis=1)
	tissue_mask_extended=np.concatenate((tissue_mask_extended, np.zeros((d0-img.shape[0], d1), dtype=np.uint8)), axis=0)
	#=====================================Create mask=====================================================
	tissue_mask=np.zeros(img.shape[0:2])
	tissue_ratio=np.add.reduceat(np.add.reduceat(tissue_mask_extended, np.arange(0, tissue_mask_extended.shape[0], patch_size), axis=0),np.arange(0, tissue_mask_extended.shape[1], patch_size), axis=1)
	tissue_ratio=tissue_ratio/(patch_size**2)
	x=[i*patch_size for i in range(tissue_ratio.shape[0])]
	x=np.repeat(x,tissue_ratio.shape[1])
	y=[i*patch_size for i in range(tissue_ratio.shape[1])]
	y=np.array(y*tissue_ratio.shape[0])
	patch_info=pd.DataFrame({"x":x, "y":y, "ratio":tissue_ratio.flatten()})
	patch_info=patch_info[patch_info["ratio"]>tissue_threshold]
	patch_info=patch_info.reset_index(drop=True)
	patches=np.zeros((patch_info.shape[0], patch_size, patch_size, 3), dtype=np.uint8) #n*patch_size*patch_size*3
	counter=0
	for _, row in patch_info.iterrows():
		x_tmp=int(row["x"])
		y_tmp=int(row["y"])
		patches[counter, :, :, :]=img_extended[x_tmp:x_tmp+patch_size,y_tmp:y_tmp+patch_size , :]
		counter+=1
	return patch_info, patches, img_extended, tissue_mask_extended


def patch_split_for_ST(img, patch_size, spot_info, x_name="pixel_x", y_name="pixel_y"):
	assert patch_size%2==0
	patches=np.zeros((spot_info.shape[0], patch_size, patch_size, 3), dtype=np.uint8) #n*patch_size*patch_size*3
	counter=0
	for _, row in spot_info.iterrows():
		x_tmp=int(row[x_name])
		y_tmp=int(row[y_name])
		patches[counter, :, :, :]=img[int(x_tmp-patch_size/2):int(x_tmp+patch_size/2),int(y_tmp-patch_size/2):int(y_tmp+patch_size/2), :]
		counter+=1
	return patches


def combine_patches(patches, patch_info, img_size0, img_size1,center=False):
	#Combine patches to WSI
	patch_size=patches.shape[2]
	d0=int(np.ceil(img_size0/patch_size)*patch_size)
	d1=int(np.ceil(img_size1/patch_size)*patch_size)
	combined=np.zeros([d0, d1, patches.shape[-1]])
	for i in range(patches.shape[0]):
		info=patch_info.iloc[i]
		x, y=int(info["x"]), int(info["y"])
		if center:
			combined[int(x-patch_size/2):int(x+patch_size/2), int(y-patch_size/2):int(y+patch_size/2)]=patches[i, ...]
		else:
			combined[int(x):int(x+patch_size), int(y):int(y+patch_size)]=patches[i, ...]
	return combined


