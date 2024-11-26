#Calculate pattern similarity
import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import matplotlib.pyplot as plt


def calculate_summary_curves(features, x, y, num_interval=100, method="mean", min_spots=5):
	#method=["median", "mean"]
	step=np.min([np.max(x)-np.min(x), np.max(y)-np.min(y)])/num_interval
	num_interval_x=math.ceil((np.max(x)-np.min(x))/step)
	num_interval_y=math.ceil((np.max(y)-np.min(y))/step)
	ret_x=np.zeros([num_interval_x, features.shape[1]])
	ret_y=np.zeros([num_interval_y, features.shape[1]])
	range_x=np.arange(np.min(x), np.max(x), step)
	range_y=np.arange(np.min(y), np.max(y), step)
	if method=="median":
		for i in range(num_interval_x):
			tmp=features[(range_x[i]<=x)&(x<=range_x[i]+step), :]
			if tmp.shape[0]>min_spots:
				ret_x[i, :]=np.median(tmp, 0)
			else:
				ret_x[i, :]=np.nan
		for i in range(num_interval_y):
			tmp=features[(range_y[i]<=y)&(y<=range_y[i]+step), :]
			if tmp.shape[0]>min_spots:
				ret_y[i, :]=np.median(tmp, 0)
			else:
				ret_y[i, :]=np.nan
	elif method=="mean":
		for i in range(num_interval_x):
			tmp=features[(range_x[i]<=x)&(x<=range_x[i]+step), :]
			if tmp.shape[0]>min_spots:
				ret_x[i, :]=np.mean(tmp, 0)
			else:
				ret_x[i, :]=np.nan
		for i in range(num_interval_y):
			tmp=features[(range_y[i]<=y)&(y<=range_y[i]+step), :]
			if tmp.shape[0]>min_spots:
				ret_y[i, :]=np.mean(tmp, 0)
			else:
				ret_y[i, :]=np.nan
	else:
		print('Method not valid. Use "median" or "mean". ')
		return 
	#Remove some intervals < min_spots
	ret_x=ret_x[~np.isnan(ret_x).any(axis=1)]
	ret_y=ret_y[~np.isnan(ret_y).any(axis=1)]
	return ret_x, ret_y

def curve_var_1D(curve1_x, curve2_x, rescale=True, two_side=False):
	assert curve1_x.shape[0]==curve2_x.shape[0]
	ret=np.zeros([curve1_x.shape[1], curve2_x.shape[1]])
	if rescale:
		curve1_x=(curve1_x-np.min(curve1_x,0))/(np.max(curve1_x,0)-np.min(curve1_x,0)+1e-7)
		curve2_x=(curve2_x-np.min(curve2_x,0))/(np.max(curve2_x,0)-np.min(curve2_x,0)+1e-7)
	if two_side:
		for i in range(curve1_x.shape[1]):
			diff1=curve2_x-curve1_x[:, [i]]
			diff2=curve2_x+curve1_x[:, [i]]
			var1=np.var(diff1, 0)
			var2=np.var(diff2, 0)
			ret[i]=np.min(np.stack((var1, var2)), 0)
	else:
		for i in range(curve1_x.shape[1]):
			diff=curve2_x-curve1_x[:, [i]]
			ret[i]=np.var(diff, 0)
	return ret

def curve_var_2D(curve1_x, curve1_y, curve2_x, curve2_y, rescale=True, two_side=False):
	var_x=curve_var_1D(curve1_x, curve2_x, rescale, two_side)
	var_y=curve_var_1D(curve1_y, curve2_y, rescale, two_side)
	w_x=curve1_x.shape[0]/(curve1_x.shape[0]+curve1_y.shape[0])
	w_y=curve1_y.shape[0]/(curve1_x.shape[0]+curve1_y.shape[0])
	return w_x*var_x+w_y*var_y

def curve_cor_1D(curve1_x, curve2_x, rescale=True, add_noise=True):
	assert curve1_x.shape[0]==curve2_x.shape[0]
	ret=np.zeros([curve1_x.shape[1], curve2_x.shape[1]])
	if rescale:
		curve1_x=(curve1_x-np.min(curve1_x,0))/(np.max(curve1_x,0)-np.min(curve1_x,0)+1e-7)
		curve2_x=(curve2_x-np.min(curve2_x,0))/(np.max(curve2_x,0)-np.min(curve2_x,0)+1e-7)
	if add_noise:
		curve1_x[0, :]+=1e-7
		curve2_x[0, :]+=1e-7
	for i in range(curve1_x.shape[1]):
		tmp=np.concatenate((curve2_x, curve1_x[:, [i]]), 1)
		tmp=np.corrcoef(tmp.T)
		ret[i]=tmp[-1][0:-1]
	return ret

def curve_cor_2D(curve1_x, curve1_y, curve2_x, curve2_y, integrate_xy="weighted", rescale=True, add_noise=True, two_side=False):
	#integrate_xy=["weighted", "min"]
	cor_x=curve_cor_1D(curve1_x, curve2_x, rescale, add_noise)
	cor_y=curve_cor_1D(curve1_y, curve2_y, rescale, add_noise)
	if two_side:
		cor_x=np.abs(cor_x)
		cor_y=np.abs(cor_y)
	if integrate_xy=="weighted":
		w_x=curve1_x.shape[0]/(curve1_x.shape[0]+curve1_y.shape[0])
		w_y=curve1_y.shape[0]/(curve1_x.shape[0]+curve1_y.shape[0])
		ret=w_x*cor_x+w_y*cor_y
	elif integrate_xy=="min":
		ret=np.min(np.abs(np.stack((cor_x, cor_y))), 0)
	else:
		print('Pooling method not valid. Use "max" or "weighted" or "equal". ')
	return ret

def curve_diff_1D(curve1_x, curve2_x, rescale=True, two_side=False):
	assert curve1_x.shape[0]==curve2_x.shape[0]
	ret=np.zeros([curve1_x.shape[1], curve2_x.shape[1]])
	if rescale:
		curve1_x=(curve1_x-np.min(curve1_x,0))/(np.max(curve1_x,0)-np.min(curve1_x,0)+1e-7)
		curve2_x=(curve2_x-np.min(curve2_x,0))/(np.max(curve2_x,0)-np.min(curve2_x,0)+1e-7)
	if two_side:
		for i in range(curve1_x.shape[1]):
			diff1=np.absolute(curve2_x-curve1_x[:, [i]])
			diff2=np.absolute(curve2_x+curve1_x[:, [i]])
			diff1=np.mean(diff1, 0)
			diff2=np.mean(diff2, 0)
			ret[i]=np.min(np.stack((diff1, diff2)), 0)
	else:
		for i in range(curve1_x.shape[1]):
			diff=np.absolute(curve2_x-curve1_x[:, [i]])
			ret[i]=np.mean(diff, 0)
	return ret

def curve_diff_2D(curve1_x, curve1_y, curve2_x, curve2_y, integrate_xy="weighted", rescale=True, two_side=False):
	#integrate_xy=["weighted", "min", "max"]
	diff_x=curve_diff_1D(curve1_x, curve2_x, rescale, two_side)
	diff_y=curve_diff_1D(curve1_y, curve2_y, rescale, two_side)
	if integrate_xy=="weighted":
		w_x=curve1_x.shape[0]/(curve1_x.shape[0]+curve1_y.shape[0])
		w_y=curve1_y.shape[0]/(curve1_x.shape[0]+curve1_y.shape[0])
		ret=w_x*diff_x+w_y*diff_y
	elif integrate_xy=="min":
		ret=np.min(np.abs(np.stack((diff_x, diff_y))), 0)
	elif integrate_xy=="max":
		ret=np.max(np.abs(np.stack((diff_x, diff_y))), 0)
	else:
		print('Pooling method not valid. Use "min" or "weighted". ')
	return ret

def pattern_similarity(df1, df2, clusters, x, y, num_interval=20, method="mean", metric="cor", integrate_xy="weighted", pool="min", rescale=True, add_noise=True, two_side=False, min_spots=5):
	#metric=["cor", "var", "diff"]
	#pool=["max", "min","weighted", "equal"]
	#integrate_xy=["weighted", "min"]
	uniq_clusters=np.unique(clusters)
	num_clusters=len(uniq_clusters)
	diff=np.zeros([num_clusters, df1.shape[1], df2.shape[1]])
	for i in range(num_clusters):
		c=uniq_clusters[i]
		sub_df1=df1[clusters==c]
		sub_df2=df2[clusters==c]
		sub_x=x[clusters==c]
		sub_y=y[clusters==c]
		curve1_x, curve1_y=calculate_summary_curves(sub_df1, sub_x, sub_y, num_interval=num_interval, method=method, min_spots=min_spots)
		curve2_x, curve2_y=calculate_summary_curves(sub_df2, sub_x, sub_y, num_interval=num_interval, method=method, min_spots=min_spots)
		if metric=="cor":
			df=curve_cor_2D(curve1_x=curve1_x, curve1_y=curve1_y, curve2_x=curve2_x, curve2_y=curve2_y, integrate_xy=integrate_xy, rescale=rescale, add_noise=add_noise, two_side=two_side)
		elif metric=="var":
			df=curve_var_2D(curve1_x=curve1_x, curve1_y=curve1_y, curve2_x=curve2_x, curve2_y=curve2_y, integrate_xy=integrate_xy, rescale=rescale, two_side=two_side)
		elif metric=="diff":
			df=curve_diff_2D(curve1_x=curve1_x, curve1_y=curve1_y, curve2_x=curve2_x, curve2_y=curve2_y, integrate_xy=integrate_xy, rescale=rescale, two_side=two_side)
		else:
			print('Metric not valid. Use "cor" or "var" or "diff". ')
			break
		diff[i]=df
	if pool=="max":
		return np.max(diff, 0)
	elif pool=="min":
		return np.min(diff, 0)
	elif pool=="equal":
		return np.mean(diff, 0)
	elif pool=="weighted":
		ret=np.zeros([df1.shape[1], df2.shape[1]])
		for i in range(num_clusters):
			c=uniq_clusters[i]
			w=np.sum(clusters==c)/len(clusters)
			ret+=diff[i]*w
		return ret
	else:
		print('Pooling method not valid. Use "max" or "weighted" or "equal". ')

def combine_clusters(clusters1, clusters2, min_threshold=1/5, max_threshold=2/3):
	df=pd.DataFrame({"clusters1":clusters1, "clusters2":clusters2})
	cross_tab=pd.crosstab(df["clusters1"], df["clusters2"])
	cross_tab=cross_tab.div(cross_tab.sum(axis=1), axis=0)
	cross_tab=(cross_tab>np.max([min_threshold, (1/cross_tab.shape[0])]))*1
	lists=[cross_tab.index[cross_tab.loc[:, i]>0].values.tolist() for i in cross_tab.columns]
	lists=[i for i in lists if len(i)>0]
	d_l={}
	d_num={}
	for i in range(len(lists)):
		d_l[i]=lists[i]
		d_num[i]=np.sum(pd.Series(clusters1).isin(lists[i]))
	#Sort by size
	keys=[k for k, v in sorted(d_num.items(), key=lambda item: -item[1])]
	ret=[]
	used=set()
	for key in keys:
		add=True
		new=d_l[key]
		new=[i for i in new if i not in used]
		used=used.union(set(new))
		for i in range(len(ret)):
			s=ret[i]
			num=np.sum(pd.Series(clusters1).isin(new))
			num_overlap=np.sum(pd.Series(clusters1).isin(s&set(new)))
			if (num_overlap/num)>max_threshold:
				ret[i]=s.union(set(new))
				add=False
				break
		if add:
			ret.append(set(new))
	label=np.zeros(len(clusters1))
	for i in range(len(ret)):
		s=ret[i]
		label+=pd.Series(clusters1).isin(s).values*i
	return label


def refine_combined_clusters(clusters, x, y, min_num=3, num_interval=100):
	uniq_cluster=np.unique(clusters)
	undefined=[]
	for i in range(len(uniq_cluster)):
		c=uniq_cluster[i]
		print("Doing ", c)
		df=pd.DataFrame({"clusters":clusters, "x":x, "y":y})
		df=df[df["clusters"]==c]
		step=np.min([np.max(df["x"])-np.min(df["x"]), np.max(df["y"])-np.min(df["y"])])/num_interval
		range_x=np.arange(np.min(df["x"]), np.max(df["x"]), step)
		range_y=np.arange(np.min(df["y"]), np.max(df["y"]), step)
		#Examine x
		for j in range(len(range_x)):
			df_tmp=df[(range_x[j]<=df["x"])&(df["x"]<=range_x[j]+step)]
			if df_tmp.shape[0]<min_num:
				undefined+=df_tmp.index.tolist()
		#Examine y
		for j in range(len(range_y)):
			df_tmp=df[(range_y[j]<=df["y"])&(df["y"]<=range_y[j]+step)]
			if df_tmp.shape[0]<min_num:
				undefined+=df_tmp.index.tolist()
	ret=[]
	df=pd.DataFrame({"clusters":clusters, "x":x, "y":y})
	for index, row in df.iterrows():
		if index in undefined:
			ret.append("undefined")
		else:
			ret.append(str(row["clusters"]))
	return ret


#green is gene, red is image
def check_pattern_similarity(df1, df2, plot_dir, clusters, x, y, num_interval=10, method="mean", metric="cor", integrate_xy="weighted", pool="min", rescale=True, add_noise=True, two_side=False):
	uniq_clusters=np.unique(clusters)
	num_clusters=len(uniq_clusters)
	ret=np.zeros([num_clusters, df1.shape[1], df2.shape[1]])
	for i in range(num_clusters):
		c=uniq_clusters[i]
		sub_df1=df1[clusters==c]
		sub_df2=df2[clusters==c]
		sub_x=x[clusters==c]
		sub_y=y[clusters==c]
		curve1_x, curve1_y=calculate_summary_curves(sub_df1, sub_x, sub_y, num_interval=num_interval, method=method, min_spots=5)
		curve2_x, curve2_y=calculate_summary_curves(sub_df2, sub_x, sub_y, num_interval=num_interval, method=method, min_spots=5)
		x_tmp=[i for i in range(curve1_x.shape[0])]
		plt.plot(x_tmp, curve1_x.flatten(), '-o', c='#15821E', markersize=5)
		plt.plot(x_tmp, curve2_x.flatten(), '-o', c='#DB4C6C', markersize=5)
		ax = plt.gca()
		ax.set_ylim([0, 1])
		plt.savefig(plot_dir+"/cluster_"+str(c)+"_x.png", dpi=300)
		plt.clf()
		x_tmp=[i for i in range(curve1_y.shape[0])]
		plt.plot(x_tmp, curve1_y.flatten(), '-o', c='#15821E', markersize=5)
		plt.plot(x_tmp, curve2_y.flatten(), '-o', c='#DB4C6C', markersize=5)
		ax = plt.gca()
		ax.set_ylim([0, 1])
		plt.savefig(plot_dir+"/cluster_"+str(c)+"_y.png", dpi=300)
		plt.clf()
		if metric=="cor":
			df=curve_cor_2D(curve1_x=curve1_x, curve1_y=curve1_y, curve2_x=curve2_x, curve2_y=curve2_y, integrate_xy=integrate_xy, rescale=rescale, add_noise=add_noise, two_side=two_side)
			print("Domain "+str(c)+" cor = "+str(df[0][0]))
		elif metric=="var":
			df=curve_var_2D(curve1_x=curve1_x, curve1_y=curve1_y, curve2_x=curve2_x, curve2_y=curve2_y, integrate_xy=integrate_xy, rescale=rescale, two_side=two_side)
			print("Domain "+str(c)+" var = "+str(df[0][0]))
		elif metric=="diff":
			df=curve_diff_2D(curve1_x=curve1_x, curve1_y=curve1_y, curve2_x=curve2_x, curve2_y=curve2_y, integrate_xy=integrate_xy, rescale=rescale, two_side=two_side)
			print("Domain "+str(c)+" diff = "+str(df[0][0]))
		else:
			print('Metric not valid. Use "cor" or "var" or "diff". ')
		ret[i]=df
	if pool=="max":
		print("max: ", np.max(ret, 0)[0][0])
	elif pool=="min":
		print("min: ", np.min(ret, 0)[0][0])
	elif pool=="equal":
		print(""+np.mean(ret, 0)[0][0])
	elif pool=="weighted":
		ret2=np.zeros([df1.shape[1], df2.shape[1]])
		for i in range(num_clusters):
			c=uniq_clusters[i]
			w=np.sum(clusters==c)/len(clusters)
			ret2+=ret[i]*w
		print("weighted: ", ret2[0][0])





"""

def combine_clusters_old(clusters1, clusters2, min_threshold=1/5):
	df=pd.DataFrame({"clusters1":clusters1, "clusters2":clusters2})
	cross_tab=pd.crosstab(df["clusters1"], df["clusters2"])
	cross_tab=cross_tab.div(cross_tab.sum(axis=1), axis=0)
	cross_tab=(cross_tab>np.max([min_threshold, (1/cross_tab.shape[0])]))*1
	lists=[cross_tab.index[cross_tab.loc[:, i]>0].values.tolist() for i in cross_tab.columns]
	ret=[]
	for l in lists:
		add=True
		for i in range(len(ret)):
			s=ret[i]
			if len(s&set(l))>0:
				ret[i]=s.union(set(l))
				add=False
		if add:
			ret.append(set(l))
	label=np.zeros(len(clusters1))
	for i in range(len(ret)):
		s=ret[i]
		label+=pd.Series(clusters1).isin(s).values*i
	return label


def curve_diff_1D(curve1_x, curve2_x, rescale=True, two_side=False):
	assert curve1_x.shape[0]==curve2_x.shape[0]
	ret=np.zeros([curve1_x.shape[1], curve2_x.shape[1]])
	if rescale:
		curve1_x=(curve1_x-np.min(curve1_x,0))/(np.max(curve1_x,0)-np.min(curve1_x,0)+1e-7)
		curve2_x=(curve2_x-np.min(curve2_x,0))/(np.max(curve2_x,0)-np.min(curve2_x,0)+1e-7)
	if two_side:
		for i in range(curve1_x.shape[1]):
			diff1=np.absolute(curve2_x-curve1_x[:, [i]])
			diff2=np.absolute(curve2_x+curve1_x[:, [i]])
			diff1=np.sum(diff1, 0)
			diff2=np.sum(diff2, 0)
			ret[i]=np.min(np.stack((diff1, diff2)), 0)
	else:
		for i in range(curve1_x.shape[1]):
			diff=np.absolute(curve2_x-curve1_x[:, [i]])
			ret[i]=np.sum(diff, 0)
	return ret

def curve_diff_2D(curve1_x, curve1_y, curve2_x, curve2_y, rescale=True, two_side=False):
	diff_x=curve_diff_1D(curve1_x, curve2_x, rescale, two_side)
	diff_y=curve_diff_1D(curve1_y, curve2_y, rescale, two_side)
	w_x=curve1_x.shape[0]/(curve1_x.shape[0]+curve1_y.shape[0])
	w_y=curve1_y.shape[0]/(curve1_x.shape[0]+curve1_y.shape[0])
	return w_x*diff_x+w_y*diff_y

def calculate_summary_curves(features, x, y, num_interval=20):
	step=np.min([np.max(x)-np.min(x), np.max(y)-np.min(y)])/num_interval
	num_interval_x=math.ceil((np.max(x)-np.min(x))/step)
	num_interval_y=math.ceil((np.max(y)-np.min(y))/step)
	ret_x=np.zeros([num_interval_x, features.shape[1]])
	ret_y=np.zeros([num_interval_y, features.shape[1]])
	range_x=np.arange(np.min(x), np.max(x), step)
	range_y=np.arange(np.min(y), np.max(y), step)
	for i in range(num_interval_x):
		tmp=(range_x[i]<=x)&(x<=range_x[i]+step)
		if np.sum(tmp)==0:
			ret_x[i, :]=0
		else:
			ret_x[i, :]=np.median(features[tmp, :], 0)
	for i in range(num_interval_y):
		tmp=(range_y[i]<=y)&(y<=range_y[i]+step)
		if np.sum(tmp)==0:
			ret_x[i, :]=0
		else:
			ret_y[i, :]=np.median(features[tmp, :], 0)
	return ret_x, ret_y
"""
