import pandas as pd
import numpy as np
import cv2
import numba
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops

@numba.njit("f4(f4[:], f4[:])")
def euclid_dist(t1,t2):
	sum=0
	for i in range(t1.shape[0]):
		sum+=(t1[i]-t2[i])**2
	return np.sqrt(sum)

@numba.njit("f4[:,:](f4[:,:])", parallel=True, nogil=True)
def pairwise_distance(X):
	n=X.shape[0]
	adj=np.empty((n, n), dtype=np.float32)
	for i in numba.prange(n):
		for j in numba.prange(i, n):
			adj[i][j]=adj[j][i]=euclid_dist(X[i], X[j])
	return adj

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

def Extract_Whole_Mask_Features_each_mask(mask):
	#Area of 1
	#Area of 1 ratio
	#Dist tranns for 0, mean, median, stdev, IQR
	#Dist tranns for 1, mean, median, stdev, IQR
	#feature_names=["Area_of_1","Area_of_1_ratio", "Dist_Trans_0_mean", "Dist_Trans_0_median", "Dist_Trans_0_std", "Dist_Trans_0_iqr", "Dist_Trans_1_mean", "Dist_Trans_1_median", "Dist_Trans_1_std", "Dist_Trans_1_iqr"]
	ret=[]
	ret.append(np.sum(mask))
	ret.append(np.sum(mask)/mask.shape[0]/mask.shape[1])
	dist0 = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
	dist1 = cv2.distanceTransform((mask==0).astype(np.uint8), cv2.DIST_L2, 3)
	ret.append(np.mean(dist0)) #Mean
	ret.append(np.quantile(dist0, 0.5)) #Median
	ret.append(np.std(dist0)) #std
	q75, q25 = np.percentile(dist0, [75 ,25])
	ret.append(q75-q25)#IQR
	ret.append(np.mean(dist1)) #Mean
	ret.append(np.quantile(dist1, 0.5)) #Median
	ret.append(np.std(dist1)) #std
	q75, q25 = np.percentile(dist1, [75 ,25])
	ret.append(q75-q25)#IQR	
	return ret

def Extract_Whole_Mask_Features(masks, patch_info):
	feature_names=["Area_of_1","Area_of_1_ratio", "Dist_Trans_0_mean", "Dist_Trans_0_median", "Dist_Trans_0_std", "Dist_Trans_0_iqr", "Dist_Trans_1_mean", "Dist_Trans_1_median", "Dist_Trans_1_std", "Dist_Trans_1_iqr"]
	names=[]
	for i in range(masks.shape[0]):
		names+=["m"+str(i)+"_"+j for j in feature_names]
	ret=np.zeros([masks.shape[1], len(names)])
	for i in range(masks.shape[1]):
		print("Doing channel ", str(i))
		ret_tmp=[]
		for j in range(masks.shape[0]):
			mask=masks[j, i, :, :] #num_masks, num_samples, size1, size2
			tmp=Extract_Whole_Mask_Features_each_mask(mask)
			ret_tmp+=tmp
		ret[i]=ret_tmp
	mask_features=np.array(ret)
	mask_features=pd.DataFrame(mask_features, index=patch_info.index.tolist(), columns=names)
	return mask_features


def Extract_CC_Features_each_CC(labels, GLCM=False, img=None):
	#Check all attr
	#[attr for attr in dir(region_props[0]) if not attr.startswith('__')]
	#labels size1 x size2
	pnames=["label","area", "bbox", "bbox_area", "convex_area", "eccentricity", "equivalent_diameter", "euler_number", 
		"extent", "filled_area", "inertia_tensor", "inertia_tensor_eigvals", "local_centroid", "major_axis_length", 
		"minor_axis_length", "orientation", "perimeter", "solidity", "centroid"]
	region_props = regionprops(labels.astype(int))
	label_area=[(prop.label, prop.area) for prop in region_props]
	ret={}
	for name in pnames:
		ret[name]=[]
		for i in range(len(region_props)):
			ret[name].append(getattr(region_props[i],name))
	ret=pd.DataFrame(ret)
	#----------------------------------
	if GLCM:
		dissimilarity, correlation=[], []
		img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		for _, row in ret.iterrows():
			bbox=row["bbox"]
			patch=img_gray[bbox[0]:bbox[2], bbox[1]:bbox[3]]
			glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True, normed=True)
			dissimilarity.append(graycoprops(glcm, 'dissimilarity')[0, 0])
			correlation.append(graycoprops(glcm, 'correlation')[0, 0])
		ret["glcm_dissimilarity"]=dissimilarity
		ret["glcm_correlation"]=correlation
	return ret


#Summarize each CC features by patch
def Extract_CC_Features(labels, patch_info, channels, min_area=0, quantiles=[0, 25, 50, 75, 100]):
	#labels: #channel x #patch x size1 x size2
	#---------------Get feature names
	names=['area', 'bbox_area', 'convex_area', 'eccentricity','equivalent_diameter',  'extent', 'filled_area','major_axis_length', 'minor_axis_length', 'orientation', 'perimeter','solidity', "hw_ratios"]
	tmp=[]
	for channel in channels:
		tmp+=["c"+str(channel)+"_"+i for i in names]
		tmp+=["c"+str(channel)+"_"+"dis"]
	columns=[]
	for name in tmp:
		columns+=[name+"_median", name+"_std", name+"_iqr"] # mean?
		columns+=[name+"_q"+str(q) for q in quantiles]
	ret=pd.DataFrame(np.zeros([patch_info.shape[0], len(columns)]), index=patch_info.index.tolist(), columns=columns)
	#---------------
	for channel in channels:
		print("Doing channel ", str(channel))
		#Get column names
		for i in range(labels.shape[1]):
			tmp=labels[channel, i, ...]
			if len(np.unique(tmp))>1:
				tmp=Extract_CC_Features_each_CC(tmp)
				#----Add features here----
				tmp["hw_ratios"]=tmp["major_axis_length"]/(tmp["minor_axis_length"]+1e-7)
				tmp=tmp[tmp["area"]>min_area]
				if tmp.shape[0]>0:
					for name in names:
						q75, q25 = np.percentile(tmp[name], [75 ,25])
						ret.iloc[i]["c"+str(channel)+"_"+name+"_iqr"]=q75-q25
						ret.iloc[i]["c"+str(channel)+"_"+name+"_median"]=np.quantile(tmp[name], 0.5)
						ret.iloc[i]["c"+str(channel)+"_"+name+"_std"]=np.std(tmp[name])
						for q in quantiles:
							ret.iloc[i]["c"+str(channel)+"_"+name+"_q"+str(q)]=np.percentile(tmp[name], q)
					#Pairwise Distance
					if tmp.shape[0]>2:
						centorids=[list(i) for i in tmp["centroid"]]
						centorids=np.array(centorids).astype(np.float32)
						dis=pairwise_distance(centorids)
						dis=dis[np.triu_indices(dis.shape[0]-1)]
						q75, q25 = np.percentile(dis, [75 ,25])
						ret.iloc[i]["c"+str(channel)+"_dis_iqr"]=q75-q25
						ret.iloc[i]["c"+str(channel)+"_dis_median"]=np.quantile(dis, 0.5)
						ret.iloc[i]["c"+str(channel)+"_dis_std"]=np.std(dis)
						for q in quantiles:
							ret.iloc[i]["c"+str(channel)+"_dis_q"+str(q)]=np.percentile(dis, q)
	return ret

def Selective_Log_Transfer(features):
	#Log or not, input as DataFrame
	selected_names=[]
	ret=features.copy()
	for name in ret.columns:
		if np.min(ret[name])>=0:
			ret["log_"+name]=np.log(ret[name]+1)
		else:
			ret["log_"+name]=np.log(ret[name]+1-np.min(ret[name]))
		# std1=np.std(ret[name])/(np.max(ret[name])-np.min(ret[name]))
		# std2=np.std(ret["log_"+name])/(np.max(ret["log_"+name])-np.min(ret["log_"+name]))
		std1=np.std(ret[name])/(np.max(ret[name])-np.min(ret[name])+1e-7)
		std2=np.std(ret["log_"+name])/(np.max(ret["log_"+name])-np.min(ret["log_"+name])+1e-7)
		if np.max([std1,std2])>0:
			selected_names.append(["log_"+name, name][std1>std2])
	ret=ret.loc[:, selected_names]
	return ret


def Log_Transfer(features):
	#Log or not, input as DataFrame
	selected_names=[]
	ret=features.copy()
	for name in ret.columns:
		if np.min(ret[name])>=0:
			ret["log_"+name]=np.log(ret[name]+1)
		else:
			ret["log_"+name]=np.log(ret[name]+1-np.min(ret[name]))
		selected_names.append("log_"+name)
	ret=ret.loc[:, selected_names]
	return ret


def radial_profile_features_each_cc(patch, binary, center=None):
	if center==None:
		center=(np.array(patch.shape)/2).astype(int)
	y, x = np.indices((patch.shape))
	r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	r = r.astype(np.int)
	r=r[binary==1].ravel()
	intensity=patch[binary==1].ravel()
	r_max=np.max(r)
	r_min=np.min(r)
	step=(r_max-r_min)/3
	index1=(r>=r_min) & (r<r_min+step)
	index2=(r>=r_min+step) & (r<r_min+step*2)
	index3=(r>=r_min+step*2)
	pixels1=intensity[index1]
	pixels2=intensity[index2]
	pixels3=intensity[index3]
	if len(pixels1)==0:
		pixels1=np.array([0])
	if len(pixels2)==0:
		pixels2=np.array([0])
	if len(pixels3)==0:
		pixels3=np.array([0])
	mean1=np.mean(pixels1)
	mean2=np.mean(pixels2)
	mean3=np.mean(pixels3)
	q75, q25 = np.percentile(pixels1, [75 ,25])
	iqr1=q75-q25
	q75, q25 = np.percentile(pixels2, [75 ,25])
	iqr2=q75-q25
	q75, q25 = np.percentile(pixels3, [75 ,25])
	iqr3=q75-q25
	std1=np.std(pixels1)
	std2=np.std(pixels2)
	std3=np.std(pixels3)
	return [mean1, mean2, mean3, iqr1, iqr2, iqr3, std1, std2, std3]

def radial_profile_features(labels, img, features, centers=None):
	#centers is a n by 2 array
	ret=np.zeros([features.shape[0], 9])
	img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	id=[]
	for i in range(ret.shape[0]):
		label=features.iloc[i]["label"]
		id.append(label)
		bbox=list(map(int, features.iloc[i]["bbox"].strip("()").split(', ')))
		patch=img_gray[bbox[0]:bbox[2], bbox[1]:bbox[3]]
		label_patch=labels[bbox[0]:bbox[2], bbox[1]:bbox[3]]
		binary=(label_patch==label)*1
		if centers==None:
			ret[i]=radial_profile_features_each_cc(patch, binary)
		else:
			center=centers[i]
			ret[i]=radial_profile_features_each_cc(patch, binary, center)
	ret=pd.DataFrame(ret)
	ret.columns=["0-1/3_mean_intensity", "1/3-2/3_mean_intensity", "2/3-1_mean_intensity", "0-1/3_iqr_intensity", "1/3-2/3_iqr_intensity", "2/3-1_iqr_intensity", "0-1/3_std_intensity", "1/3-2/3_std_intensity", "2/3-1_std_intensity"]
	ret["label"]=id
	return ret



"""
def Extract_CC_Features(labels, patch_info, channels, min_area=0):
	#---------------Get feature names
	names=['area', 'bbox_area', 'convex_area', 'eccentricity','equivalent_diameter',  'extent', 'filled_area','major_axis_length', 'minor_axis_length', 'orientation', 'perimeter','solidity']
	tmp=[]
	for channel in channels:
		tmp+=["c"+str(channel)+"_"+i for i in names]
		tmp+=["c"+str(channel)+"_"+"dis"]
	columns=[]
	for name in tmp:
		columns+=[name+"_median", name+"_std", name+"_iqr"]
	ret=pd.DataFrame(np.zeros([patch_info.shape[0], len(columns)]), index=patch_info.index.tolist(), columns=columns)
	#---------------
	for channel in channels:
		print("Doing channel ", str(channel))
		#Get column names
		for i in range(labels.shape[1]):
			tmp=labels[channel, i, ...]
			if len(np.unique(tmp))>1:
				tmp=Extract_CC_Features_each_CC(tmp)
				tmp=tmp[tmp["area"]>min_area]
				if tmp.shape[0]>0:
					for name in names:
						q75, q25 = np.percentile(tmp[name], [75 ,25])
						ret.iloc[i]["c"+str(channel)+"_"+name+"_iqr"]=q75-q25
						ret.iloc[i]["c"+str(channel)+"_"+name+"_median"]=np.quantile(tmp[name], 0.5)
						ret.iloc[i]["c"+str(channel)+"_"+name+"_std"]=np.std(tmp[name])
					#Pairwise Distance
					if tmp.shape[0]>2:
						centorids=[list(i) for i in tmp["centroid"]]
						centorids=np.array(centorids).astype(np.float32)
						dis=pairwise_distance(centorids)
						dis=dis[np.triu_indices(dis.shape[0]-1)]
						q75, q25 = np.percentile(dis, [75 ,25])
						ret.iloc[i]["c"+str(channel)+"_dis_iqr"]=q75-q25
						ret.iloc[i]["c"+str(channel)+"_dis_median"]=np.quantile(dis, 0.5)
						ret.iloc[i]["c"+str(channel)+"_dis_std"]=np.std(dis)
	return ret



def Extract_CC_Features_each_CC(labels_combined, pre_filtered_cc_index=[]):
	#num_cc, cc_areas, box_areas, area_ratios, hw_ratios, cc_min_dis
	cc_index=np.unique(labels_combined.astype("int")).tolist()
	#If have pre filtered results
	if len(pre_filtered_cc_index)>0:
		cc_index=[i for i in cc_index if i in pre_filtered_cc_index]
	if len(cc_index)>0:
		label_area, cc_areas, box_areas, area_ratios, hw_ratios, cc_coords=[], [], [], [], [], []
		for i in range(len(cc_index)):
			label=cc_index[i]
			cc=np.array(labels_combined==label, dtype=np.uint8)
			label_area.append(np.sum(cc))
			# Find contours in img.
			cnts = cv2.findContours(cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  
			# Find the contour with the maximum area.
			c = max(cnts, key=cv2.contourArea)
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			#_=cv2.drawContours(ret, [box], 0, ( 255), 1)
			#cv2.imwrite('./figures/t2.jpg', ret)
			w, h=np.linalg.norm(box[0] - box[1])+1, np.linalg.norm(box[0] - box[3])+1
			w, h=np.min([w, h]), np.max([w, h])
			center=np.mean(box, 0).astype("int")
			cc_area=np.sum(cc)
			box_area=w*h
			area_ratio=cc_area/box_area
			hw_ratio=h/w
			cc_coord=[center[1], center[0]]
			cc_areas.append(cc_area)
			box_areas.append(box_area)
			area_ratios.append(area_ratio)
			hw_ratios.append(hw_ratio)
			cc_coords.append(cc_coord)
		cc_coords=np.array(cc_coords).astype("float32")
		cc_dis=pairwise_distance(cc_coords)
		np.fill_diagonal(cc_dis, labels_combined.shape[0]*2)
		cc_min_dis=np.min(cc_dis, 0)
		ret=pd.DataFrame({'cc_index': cc_index,'cc_areas': cc_areas,'area_ratios': area_ratios,'hw_ratios': hw_ratios,'cc_coords': cc_coords.tolist()})
		return ret
	else:
		print("No CC found!")

def Extract_CC_Features(labels_combined,  quantiles=[0, 0.25, 0.5, 0.75, 1]):
	#num_cc, cc_areas, box_areas, area_ratios, hw_ratios, cc_min_dis
	label_area=[]
	for i in np.unique(labels_combined):
		label_area.append((int(i), np.sum(labels_combined==i)))
	ret=[len(label_area)]
	if len(label_area)>0:
		cc_areas, box_areas, area_ratios, hw_ratios, cc_coords=[], [], [], [], []
		for i in range(len(label_area)):
			label=label_area[i][0]
			cc=np.array(labels_combined==label, dtype=np.uint8)
			# Find contours in img.
			cnts = cv2.findContours(cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  
			# Find the contour with the maximum area.
			c = max(cnts, key=cv2.contourArea)
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			#_=cv2.drawContours(ret, [box], 0, ( 255), 1)
			#cv2.imwrite('./figures/t2.jpg', ret)
			w, h=np.linalg.norm(box[0] - box[1])+1, np.linalg.norm(box[0] - box[3])+1
			w, h=np.min([w, h]), np.max([w, h])
			center=np.mean(box, 0).astype("int")
			cc_area=np.sum(cc)
			box_area=w*h
			area_ratio=cc_area/box_area
			hw_ratio=h/w
			cc_coord=[center[1], center[0]]
			cc_areas.append(cc_area)
			box_areas.append(box_area)
			area_ratios.append(area_ratio)
			hw_ratios.append(hw_ratio)
			cc_coords.append(cc_coord)
		cc_coords=np.array(cc_coords).astype("float32")
		cc_dis=pairwise_distance(cc_coords)
		np.fill_diagonal(cc_dis, labels_combined.shape[0]*2)
		cc_min_dis=np.min(cc_dis, 0)
		for q in quantiles:
			ret.append(np.quantile(cc_areas, q))
		ret.append(np.var(cc_areas))
		for q in quantiles:
			ret.append(np.quantile(box_areas, q))
		ret.append(np.var(box_areas))
		for q in quantiles:
			ret.append(np.quantile(area_ratios, q))
		ret.append(np.var(area_ratios))
		for q in quantiles:
			ret.append(np.quantile(hw_ratios, q))
		ret.append(np.var(hw_ratios))
		for q in quantiles:
			ret.append(np.quantile(cc_min_dis, q))
		ret.append(np.var(cc_min_dis))
	else:
		ret+=[0]*(5*(len(quantiles)+1))
	return ret


def Extract_CC_Features(mask, min_area=20, connectivity=4, quantiles=[0, 0.25, 0.5, 0.75, 1]):
	#num_cc, cc_areas, box_areas, area_ratios, hw_ratios, cc_min_dis
	label_area, labels=Connected_Components_Separation(mask, connectivity=connectivity, min_area=min_area)
	ret=[len(label_area)]
	if len(label_area)>0:
		cc_areas, box_areas, area_ratios, hw_ratios, cc_coords=[], [], [], [], []
		for i in range(len(label_area)):
			label=label_area[i][0]
			cc=np.array(labels==label, dtype=np.uint8)
			# Find contours in img.
			cnts = cv2.findContours(cc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  
			# Find the contour with the maximum area.
			c = max(cnts, key=cv2.contourArea)
			rect = cv2.minAreaRect(c)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			#_=cv2.drawContours(ret, [box], 0, ( 255), 1)
			#cv2.imwrite('./figures/t2.jpg', ret)
			w, h=np.linalg.norm(box[0] - box[1])+1, np.linalg.norm(box[0] - box[3])+1
			w, h=np.min([w, h]), np.max([w, h])
			center=np.mean(box, 0).astype("int")
			cc_area=np.sum(cc)
			box_area=w*h
			area_ratio=cc_area/box_area
			hw_ratio=h/w
			cc_coord=[center[1], center[0]]
			cc_areas.append(cc_area)
			box_areas.append(box_area)
			area_ratios.append(area_ratio)
			hw_ratios.append(hw_ratio)
			cc_coords.append(cc_coord)
		cc_coords=np.array(cc_coords).astype("float32")
		cc_dis=pairwise_distance(cc_coords)
		np.fill_diagonal(cc_dis, mask.shape[0]*2)
		cc_min_dis=np.min(cc_dis, 0)
		for q in quantiles:
			ret.append(np.quantile(cc_areas, q))
		ret.append(np.var(cc_areas))
		for q in quantiles:
			ret.append(np.quantile(box_areas, q))
		ret.append(np.var(box_areas))
		for q in quantiles:
			ret.append(np.quantile(area_ratios, q))
		ret.append(np.var(area_ratios))
		for q in quantiles:
			ret.append(np.quantile(hw_ratios, q))
		ret.append(np.var(hw_ratios))
		for q in quantiles:
			ret.append(np.quantile(cc_min_dis, q))
		ret.append(np.var(cc_min_dis))
	else:
		ret+=[0]*(5*(len(quantiles)+1))
	return ret
"""