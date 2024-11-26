# Tutorial relevant functions
import os,csv,time,re,pickle,argparse
import numpy as np
import pandas as pd
import cv2
from scipy.sparse import issparse
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Local import
from .pattern_similarity import *


# Set colors
cnt_color=clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370',"#EAE7CC","#FD1593"], N=256)
cat_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]


#---------------------------------------------- Functions ----------------------------------------------
# genes_set: specify a set of genes (or from DE analysis) that are related to the interested biological process
# mask_channel: interested mask channel
# testified
def cpsi_eva(input_gene_adata, input_img_adata, genes_set, mask_channel, x_col="pixel_x", y_col="pixel_y", w_cor=0.5):
	gene_adata_sub=input_gene_adata[:,input_gene_adata.var.index.isin(genes_set)].copy()
	features=[i for i in input_img_adata.var.index if "m"+str(mask_channel) in i]+[i for i in input_img_adata.var.index if "c"+str(mask_channel) in i]
	img_adata_sub=input_img_adata[:,input_img_adata.var.index.isin(features)].copy()
	# Normalize gene expression and image features to the value range of [0,1]
	# gene expression
	gene_df=gene_adata_sub.X.A if issparse(gene_adata_sub.X) else gene_adata_sub.X
	gene_df=np.array(gene_df)
	gene_df=(gene_df-np.min(gene_df, 0))/(np.max(gene_df, 0)-np.min(gene_df, 0))
	# image features
	img_df=img_adata_sub.X.A if issparse(img_adata_sub.X) else img_adata_sub.X
	img_df=np.array(img_df)
	img_df=(img_df-np.min(img_df, 0))/(np.max(img_df, 0)-np.min(img_df, 0))
	# spatial coordinates of spots
	x=gene_adata_sub.obs[x_col].values
	y=gene_adata_sub.obs[y_col].values
	# Measure the regional pattern similarity
	clusters=[0]*len(x)
	cor=pattern_similarity(gene_df, img_df, clusters, x, y, num_interval=20, method="mean", metric="cor", integrate_xy="weighted",pool="min", rescale=True, add_noise=True, two_side=False, min_spots=5)
	diff=pattern_similarity(gene_df, img_df, clusters, x, y, num_interval=20, method="mean", metric="diff", integrate_xy="weighted",pool="max", rescale=True, add_noise=True, two_side=False, min_spots=5)
	cor=pd.DataFrame(cor, index=gene_adata_sub.var.index, columns=img_adata_sub.var.index)
	diff=pd.DataFrame(diff, index=gene_adata_sub.var.index, columns=img_adata_sub.var.index)
	# assign weights to correlation (default value is 0.5)
	CPSI=w_cor*cor+(1-w_cor)*(1-diff) # CPSI is a data frame with rows corresponding to the selected set of genes and columns corresponding to generated image features in the target mask channel
	print(CPSI.head()) # the regional similarity between the specified gene expression features and extracted morphplogical features
	return CPSI


# identify the image feature that has the highest CPSI with the target gene and generate a gradient marginal curve along x-axis and y-axis for the target pair of gene expression and image feature
# testified
def marginal_curve(input_gene_adata, input_img_adata, CPSI, g, range_step=0.25, num_cuts=5):
	# Identify the image feature that has the highest CPSI with the target gene
	f=CPSI.loc[g,:].nlargest(1).index.tolist()[0]
	print("The identified image feature having the highest CPSI with the target gene "+g+": "+f)
	# Gradient changes along x-axis and y-axis
	gene_adata_sub=input_gene_adata[:,input_gene_adata.var.index==g].copy()
	img_adata_sub=input_img_adata[:,input_img_adata.var.index==f].copy()
	img_adata_sub.obs[g]=np.array(gene_adata_sub.X)[:, gene_adata_sub.var.index==g]
	img_adata_sub.obs[f]=np.array(img_adata_sub.X)[:, img_adata_sub.var.index==f]
	x, y, z=[], [], []
	for i in range(num_cuts):
		mx=np.quantile(img_adata_sub.obs[f], i/num_cuts+1/num_cuts*(1-range_step)) # image feature: each cut range 75%
		mi=np.quantile(img_adata_sub.obs[f], i/num_cuts+1/num_cuts*range_step) # image feature: each cut range 25%
		sub_tmp=img_adata_sub[(img_adata_sub.obs[f]>=mi)&(img_adata_sub.obs[f]<=mx),:] # remain the spots with image feature falling in each cut range 25% - 75%
		median_f=np.median(sub_tmp.obs[f])
		samples=sub_tmp.obs.index[(sub_tmp.obs[f]>=mi) & (sub_tmp.obs[f]<=mx)].tolist()
		x.append(np.round(np.mean(sub_tmp.obs[f]), 3)) # calculate the mean image feature level
		y.append(np.round(np.mean(sub_tmp.obs[g]), 3)) # calculate the mean gene expression level
		z.append(np.round(np.mean(sub_tmp.obs[f]), 3))
		z.append(np.round(np.mean(sub_tmp.obs[g]), 3))
	return f, x, y, z


