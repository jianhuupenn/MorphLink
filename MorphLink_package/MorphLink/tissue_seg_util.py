import pandas as pd
import numpy as np
import cv2


def filter_contours(contours, hierarchy, filter_params):
	filtered = []
	# find indices of foreground contours (parent == -1)
	hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
	all_holes = []	
	# loop through foreground contour indices
	for cont_idx in hierarchy_1:
		# actual contour
		cont = contours[cont_idx]
		# indices of holes contained in this contour (children of parent contour)
		holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
		# take contour area (includes holes)
		a = cv2.contourArea(cont)
		# calculate the contour area of each hole
		hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
		# actual area of foreground contour region
		a = a - np.array(hole_areas).sum()
		if a == 0: continue
		if tuple((filter_params['a_t'],)) < tuple((a,)): 
			filtered.append(cont_idx)
			all_holes.append(holes)
	foreground_contours = [contours[cont_idx] for cont_idx in filtered]
	hole_contours = []
	for hole_ids in all_holes:
		unfiltered_holes = [contours[idx] for idx in hole_ids ]
		unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
		# take max_n_holes largest holes by area
		unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
		filtered_holes = []
		# filter these holes
		for hole in unfilered_holes:
			if cv2.contourArea(hole) > filter_params['a_h']:
				filtered_holes.append(hole)
		hole_contours.append(filtered_holes)
	return foreground_contours, hole_contours

def scaleContourDim(contours, scale):
	return [np.array(cont * scale, dtype='int32') for cont in contours]

def scaleHolesDim(contours, scale):
	return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

def tissu_segmentation(img,
	filter_params = {'a_t':100, 'a_h': 5, 'max_n_holes':10}, # a_t: area of tissue; a_h: area of hole;Both measured by number of patched
	ref_patch_size=512,
	sthresh=20,
	sthresh_up = 255,
	mthresh=7,
	close = 0,
	use_otsu=False,
	scale = [1.0, 1.0]):
	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
	img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
	# Thresholding
	if use_otsu:
		_, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
	else:
		_, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
	# Morphological closing
	if close > 0:
		kernel = np.ones((close, close), np.uint8)
		img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
	scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
	filter_params = filter_params.copy()
	filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
	filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area
	# Find and filter contours
	contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
	hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
	if filter_params: 
		foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
	contours_tissue = scaleContourDim(foreground_contours, scale)
	holes_tissue = scaleHolesDim(hole_contours, scale)
	#contours_tissue is a list of cnts
	#holes_tissue is a list of lists of cnts
	return contours_tissue, holes_tissue


