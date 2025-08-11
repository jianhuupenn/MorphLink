# Tutorial relevant functions
import os,csv,time,re,pickle,argparse
import numpy as np
import pandas as pd
import cv2
import scanpy as sc
import scipy
from scipy.sparse import issparse
from sklearn.decomposition import PCA
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import gaussian_kde
from .pattern_similarity import pattern_similarity

# Set colors
cnt_color=clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370',"#EAE7CC","#FD1593"], N=256)
cat_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]


#---------------------------------------------- Functions ----------------------------------------------
# test patch size
# testified
def test_patch_size(input_img, patch_size, pixel_x, pixel_y, plot_dir):
	half_size=patch_size/2
	img_new=input_img.copy()
	for i in range(len(pixel_x)):
		x=pixel_x[i]
		y=pixel_y[i]
		img_new[int(x-half_size):int(x+half_size), int(y-half_size):int(y+half_size),:]=0
	img_new=cv2.resize(img_new, (2000, 2000), interpolation = cv2.INTER_AREA)
	img_new_cvt=cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
	plt.imshow(img_new_cvt)
	plt.show()
	# save the test patch size image
	cv2.imwrite(plot_dir+'/figures/test_patch_size.jpg', img_new)


# testified
def louvain_clustering(input_adata, pca_num, n_neighbors, resolution, pred_key):
	# Reduce dimensions by PCA
	pca = PCA(n_components=pca_num)
	pca.fit(input_adata.X)
	embed=pca.transform(input_adata.X)
	tmp=sc.AnnData(embed)
	# Leiden clustering
	sc.pp.neighbors(tmp, n_neighbors=n_neighbors)
	sc.tl.leiden(tmp,resolution=resolution)
	y_pred=tmp.obs['leiden'].astype(int).to_numpy()
	input_adata.obs[pred_key]=y_pred
	input_adata.obs[pred_key]=input_adata.obs[pred_key].astype("category")
	return input_adata


# testified
def cat_figure(input_adata, x_col, y_col, color_key, color_map, plot_dir):
	ax=sc.pl.scatter(input_adata,alpha=1,x=x_col,y=y_col,color=color_key,title=color_key,color_map=color_map,show=False,size=150000/input_adata.shape[0])
	ax.set_aspect('equal', 'box')
	ax.axes.invert_yaxis()
	plt.savefig(plot_dir+"/figures/gene_pred.png", dpi=300)
	plt.show()
	plt.close()
	plt.clf()


# target_f_scores: CPSIs between the selected image feature and the target set of genes
# other_f_scores: CPSIs between other image features and the target set of genes
# testified
def cpsi_distri_histo(target_f_scores, other_f_scores):
	plt.hist(target_f_scores, bins=30, color="#ff7f0e", alpha=0.55, edgecolor="white", density=True, label="Target") # target image feature
	plt.hist(other_f_scores, bins=30, color="#1f77b4", alpha=0.55, edgecolor="white", density=True, label="Other") # other image features
	target_kde=gaussian_kde(target_f_scores)
	other_kde=gaussian_kde(other_f_scores)
	x_vals=np.linspace(min(target_f_scores.min(), other_f_scores.min()), max(target_f_scores.max(), other_f_scores.max()), 500)
	plt.plot(x_vals, target_kde(x_vals), color='grey', linewidth=1, alpha=0.5)
	plt.plot(x_vals, other_kde(x_vals), color='grey', linewidth=1, alpha=0.5)
	plt.title("The distribution of CPSIs between genes set and image features", fontsize=16)
	plt.xlabel("CPSI", fontsize=14)
	plt.ylabel("Frequency", fontsize=14)
	plt.legend(fontsize=14)
	plt.show()
	plt.close()
	plt.clf()


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


# generate sample linkage visual demonstration
# testified
def sample_linkage_visualization(visual_img_list, target_features, arrow_img, plot_dir):
	for i in range(len(visual_img_list)):
	    f=target_features[i]
	    visual_img=visual_img_list[i]
	    visual_img_cvt=cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB)
	    # resize the arrow image to match the width (have some issues here)
	    arrow_img_rz=cv2.resize(arrow_img, (visual_img_cvt.shape[1], 250), interpolation=cv2.INTER_AREA) # arrow_height = 250
	    # creat an gap
	    gap_img=(np.ones((100, visual_img_cvt.shape[1],3))*255).astype(np.uint8) # gap_height = 100
	    # combine the images vertically
	    combined_img=np.vstack((visual_img_cvt, gap_img, arrow_img_rz))
	    # plot the combined image
	    plt.figure(figsize=(15,35))
	    plt.imshow(combined_img)
	    plt.axis('off')
	    plt.savefig(plot_dir+"/figures/linkage_demonstration_"+f+"+arrow.png", dpi=300, bbox_inches='tight', pad_inches=0.8)
	    plt.show()
	    plt.close()
	    plt.clf()


