#For Kmeans clustering
from .convolve2D import convolve2D_clean, convolve2D_refine, convolve2D_nbr_num
import numpy as np
import scipy.stats

def refine_labels(pred, patch_size, refine_threshold):
	clusters=pred.reshape(patch_size, patch_size)
	# pred_refined=convolve2D_refine(clusters, kernel=[[1,1,1], [1,1,1], [1,1,1]], padding=1, strides=1, padding_value=scipy.stats.mode(clusters, axis=None).mode[0], threshold=refine_threshold)
	pred_refined=convolve2D_refine(clusters, kernel=[[1,1,1], [1,1,1], [1,1,1]], padding=1, strides=1, padding_value=scipy.stats.mode(clusters, axis=None).mode, threshold=refine_threshold)
	pred_refined=pred_refined.flatten().astype("int32")
	return pred_refined



def merge_labels(pred_refined, img, patch_size, merge_threshold, color_diff="max"):
	#Calculate cluster area, median color
	clusters=pred_refined.reshape(patch_size, patch_size)
	cs=list(np.unique(clusters))
	c_a=[(j, np.sum(pred_refined==j)/len(pred_refined)) for j in np.unique(pred_refined)]
	c_a=sorted(c_a, key=lambda x: x[1], reverse=True)
	c_a=dict((x, y) for x, y in c_a)
	c_m={} #cluster_marker expression
	for j in np.unique(pred_refined):
		c_m[j]=np.median(img[clusters==j], 0).tolist()
	pred_merged=pred_refined.copy()
	c_unseen=set(list(c_m.keys()))
	c_seen=set()
	for c0 in list(c_m.keys()):
		if c0 not in c_seen:
			c_seen.add(c0)
			c_unseen.remove(c0)
			for c1 in c_unseen:
				if color_diff=="max":
					diff=np.max(np.abs(np.array(c_m[c0])-np.array(c_m[c1])))
				elif color_diff=="distance":
					diff=np.sqrt(np.sum((np.array(c_m[c0])-np.array(c_m[c1]))**2))
				else:
					print('color_diff not understood, please use "max" or "distance".')
				if diff<=merge_threshold:
					#if adj.loc[c1, c0] > (c_a[c1]):#look at c1's nbr
					print("Diff=",diff,", average nbr=",np.round(np.mean(nbr[clusters==c1]), 2),", merging ", c1," to " ,c0)
					pred_merged[pred_merged==c1]=c0
	return pred_merged



"""
def merge_labels(pred_refined, img, patch_size, merge_threshold, average_nbr_threshold, color_diff="max"):
	#Calculate cluster area, median color
	clusters=pred_refined.reshape(patch_size, patch_size)
	cs=list(np.unique(clusters))
	c_a=[(j, np.sum(pred_refined==j)/len(pred_refined)) for j in np.unique(pred_refined)]
	c_a=sorted(c_a, key=lambda x: x[1], reverse=True)
	c_a=dict((x, y) for x, y in c_a)
	c_m={} #cluster_marker expression
	for j in np.unique(pred_refined):
		c_m[j]=np.median(img[clusters==j], 0).tolist()
	pred_merged=pred_refined.copy()
	c_unseen=set(list(c_m.keys()))
	c_seen=set()
	for c0 in list(c_m.keys()):
		if c0 not in c_seen:
			c_seen.add(c0)
			c_unseen.remove(c0)
			for c1 in c_unseen:
				if color_diff=="max":
					diff=np.max(np.abs(np.array(c_m[c0])-np.array(c_m[c1])))
				elif color_diff=="distance":
					diff=np.sqrt(np.sum((np.array(c_m[c0])-np.array(c_m[c1]))**2))
				else:
					print('color_diff not understood, please use "max" or "distance".')
				if diff<=merge_threshold:
					#if adj.loc[c1, c0] > (c_a[c1]):#look at c1's nbr
					nbr=convolve2D_nbr_num(image=clusters, center=c1, nbr=c0)
					if np.mean(nbr[clusters==c1])>=average_nbr_threshold:
						print("Diff=",diff,", average nbr=",np.round(np.mean(nbr[clusters==c1]), 2),", merging ", c1," to " ,c0)
						pred_merged[pred_merged==c1]=c0
	return pred_merged

"""
