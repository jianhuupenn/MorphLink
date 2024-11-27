<h1><center>MorphLink Tutorial</center></h1>


<center>Author: Jing Huang, Chenyang Yuan, Jiahui Jiang, Jianfeng Chen, Sunil S. Badve, Yesim Gokmen-Polar, Rossana L. Segura, Xinmiao Yan, Alexander Lazar, Jianjun Gao, Bing Yao, Michael Epstein, Linghua Wang* and Jian Hu*

### Outline
1. [Installation](#1-installation)
2. [Import modules](#2-import-python-modules)
3. [Read in data](#3-read-in-data)
4. [Image segmentation](#4-image-segmentation)
5. [Extract interpretable image features](#5-extract-interpretable-image-features)
6. [Link image features with gene expression](#6-link-image-features-with-gene-expression)
7. [Select samples for visual demonstration](#7-select-samples-for-visual-demonstration)
8. [Parameter settings](#8-parameter-settings)

### 1. Installation
To install MorphLink package you must make sure that your python version is over 3.5. If you don’t know the version of python you can check it by:


```python
import platform
platform.python_version()
```

Note: Because MorphLink depends on pytorch, you should make sure that torch is correctly installed.
<br>
Now you can install the current release of MorphLink by the following three ways:
#### 1.1 PyPI: Directly install the package from PyPI.


```python
pip3 install MorphLink
# Note: you need to make sure that the pip is for python3

# or we could install MorphLink by
python3 -m pip install MorphLink

# If you do not have permission (when you get a permission denied error), you should install MorphLink by
pip3 install --user MorphLink
```

#### 1.2 Github
Download the package from Github and install it locally:


```python
git clone https://github.com/jianhuupenn/MorphLink
cd MorphLink/MorphLink_package/
python3 setup.py install --user
```

#### 1.3 Anaconda
If you do not have Python3.5 or Python3.6 installed, consider installing Anaconda (see Installing Anaconda). After installing Anaconda, you can create a new environment, for example, MorphLink_env (or any name that you like).


```python
# create an environment called MorphLink_env
conda create -n MorphLink_env python=3.7.9

# activate your environment 
conda activate MorphLink_env
git clone https://github.com/jianhuupenn/MorphLink
cd MorphLink/MorphLink_package/
python3 setup.py build
python3 setup.py install
conda deactivate
```

### 2. Import python modules


```python
import os,csv,time,re,pickle,argparse
import numpy as np
import pandas as pd
import math
import random
import numba
import leidenalg
import anndata as ad
from anndata import AnnData,read_csv,read_text,read_mtx
import scipy
from scipy import stats
from scipy import ndimage
from scipy.sparse import issparse
from scipy.stats import gaussian_kde
import scanpy as sc
# import SpaGCN as spg
import MorphLink as mph
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage import io
from skimage.util import img_as_ubyte
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops, peak_local_max
from skimage.segmentation import watershed
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import seaborn as sns
import imutils
import cv2
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40)) # os system settings
import warnings
warnings.filterwarnings("ignore")

```


```python
mph.__version__
```




    '1.0.4'


### 3. Read in data
Data notes:
- Toydata are made available at the [shared folder](https://drive.google.com/drive/folders/1NgJICg1jFD2HP7WGZ9vXk7GrRJRoFfSD?usp=sharing).
- We have also provided with pre-generated image features in the [results folder](https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/results) to allow users to directly explore the linkage between generated image features and interested genes set, as detailed in [6. Link image features with gene expression](#6-link-image-features-with-gene-expression) and [7. Select samples for visual demonstration](#7-select-samples-for-visual-demonstration).
	For pre-generated image features:
	- mask_features_all_logged.h5ad: mask-level image features;
	- cc_features_all_logges.h5ad: object-level image features;
	- all_features_logged.h5ad: final generated image features combining both mask-level image features and object-level image features.
- The intermediate results are saved in the [figures folder](https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures).
<br>


The current version of MorphLink requres two input data: 
1. The gene expression matrix (N $\times$ G): expression_matrix.h5ad;
The gene expression data here is stored as an AnnData object. AnnData stores a data matrix .X together with annotations of observations .obs, variables .var and unstructured annotations .uns.
2. Histology image: histology.jpg (can be .tif or .png or .jpg).


```python
# Set the working directory
plot_dir="." # set a data directory (need to make up folders of results, seg_results, and figures)
if not os.path.exists(plot_dir+"/figures"):
	os.mkdir(plot_dir+"/figures")


if not os.path.exists(plot_dir+"/results"):
	os.mkdir(plot_dir+"/results")


if not os.path.exists(plot_dir+"/seg_results"):
	os.mkdir(plot_dir+"/seg_results")

    

```


```python
# Read in gene expression adata
gene_adata=sc.read("./toy_data/exp_tumor.h5ad")

# Read in histology image
img=cv2.imread("./toy_data/img_tumor.jpg")
d0, d1=img.shape[0], img.shape[1]

```

### 4. Image segmentation

#### 4.1 Determine patch size

- patch_size varies with datasets generated from different techniques (default value for 10x Visium is 400)


```python
# Set the patch size
patch_size=400
half_size=patch_size/2

# spatial coordinates of spots
pixel_x=gene_adata.obs["pixel_x"].tolist()
pixel_y=gene_adata.obs["pixel_y"].tolist()

# Test the patch size 
img_new=img.copy()
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

```
<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/test_patch_size.jpg" width=35% height=35%>


#### 4.2 Patch split

- patches: a 4D array with a shape of (N, m, m, 3), where N stands for the total number of spots and m denotes the specified patch size


```python
patches=mph.patch_split_for_ST(img=img, patch_size=patch_size, spot_info=gene_adata.obs, x_name="pixel_x", y_name="pixel_y")
# spot information
patch_info=gene_adata.obs 
patch_info["x"]=patch_info["pixel_x"]
patch_info["y"]=patch_info["pixel_y"]

# Save the splitted image patches and its patch_info
patch_info.to_csv(plot_dir+"/results/patch_info.csv")
np.save(plot_dir+"/results/patches.npy", patches)

```


```python
patches=np.load(plot_dir+"/results/patches.npy")
patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
```

#### 4.3 Segment each patch into masks

- n_clusters: equals to the number of masks within each patch (default value is 10) 
- refine the initial K-Means clusters by a convolution layer


```python
# Perform a K-Means clustering to divide the pixels of each image patch into clusters 
# then employ a convolution layer to refine the cluster assignment
mph.step4_Segmentation(plot_dir=plot_dir, n_clusters=10, refine=True, refine_threshold=4) # take around 2h
mph.check_dic_list(plot_dir)

```

#### 4.4 Match masks across patches

- num_mask_each: the number of masks within each patch (default value is 10)
- mapping_threshold1: max single color channel difference, choose all channels
- mapping_threshold2: max single color channel difference, choose one channel


```python
# Identify shared clusters across patches based on color distance
num_mask_each=10
mapping_threshold1=30  # max single color channel difference, choose all channels
mapping_threshold2=60  # max single color channel difference, choose one channel
masks, masks_index=mph.step5_Extract_Masks(plot_dir=plot_dir, patch_size=patch_size, num_mask_each=num_mask_each, mapping_threshold1=mapping_threshold1, mapping_threshold2=mapping_threshold2)

# Plot the segmentated masks
mph.step6_Plot_Masks(plot_dir=plot_dir, d0=d0, d1=d1, masks=masks, patch_size=patch_size, mapping_threshold1=mapping_threshold1, mapping_threshold2=mapping_threshold2)

```

### 5. Extract interpretable image features

#### 5.1 Mask-level image features


```python
num_mask_each=10
mapping_threshold1=30  # max single color channel difference, choose all channels
mapping_threshold2=60 
masks=np.load(plot_dir+"/results/masks_"+str(mapping_threshold1)+"_"+str(mapping_threshold2)+".npy")
with open(plot_dir+"/results/masks_index_"+str(mapping_threshold1)+"_"+str(mapping_threshold2)+".pkl", "rb") as f:
	masks_index = pickle.load(f)

```


```python
ret=mph.Extract_Whole_Mask_Features(masks, patch_info)
ret_logged=mph.Selective_Log_Transfer(ret)

```


```python
# print(ret_logged.head()) # mask-level image features
```


```python
# Save the extracted mask-level image features
ret=sc.AnnData(ret.values,obs=patch_info, var=pd.DataFrame({"feature_names":ret.columns.tolist()}))
ret.var.index=ret.var["feature_names"].tolist()
ret_logged=sc.AnnData(ret_logged.values,obs=patch_info, var=pd.DataFrame({"feature_names":ret_logged.columns.tolist()}))
ret_logged.var.index=ret_logged.var["feature_names"].tolist()
ret_logged.write_h5ad(plot_dir+"/results/mask_features_all_logged.h5ad")

```

#### 5.2 Object-level image features


```python
# Separate the connected components within each mask
mph.step8_CC_Detection_for_ST(plot_dir=plot_dir, patch_info=patch_info, masks_selected=masks, masks_index_selected=masks_index, details=False)

# Summarize image features for connected components by patch
labels=np.load(plot_dir+"/results/cc_no_details.npy")
channels=[i for i in range(labels.shape[0])]
ret=mph.Extract_CC_Features(labels=labels, patch_info=patch_info, channels=channels, min_area=10)
ret_logged=mph.Selective_Log_Transfer(ret)

```


```python
# print(ret_logged.head()) # object-level image features
```


```python
# Save the extracted object-level image features
ret=sc.AnnData(ret.values,obs=patch_info, var=pd.DataFrame({"feature_names":ret.columns.tolist()}))
ret.var.index=ret.var["feature_names"].tolist()
ret_logged=sc.AnnData(ret_logged.values,obs=patch_info, var=pd.DataFrame({"feature_names":ret_logged.columns.tolist()}))
ret_logged.var.index=ret_logged.var["feature_names"].tolist()
ret_logged.write_h5ad(plot_dir+"/results/cc_features_all_logged.h5ad")

# Combine mask-level image features with object-level image features
sub1=sc.read(plot_dir+"/results/mask_features_all_logged.h5ad")
sub2=sc.read(plot_dir+"/results/cc_features_all_logged.h5ad")
img_adata=ad.concat([sub1, sub2], axis=1,join='inner')
img_adata.obs=sub1.obs
del sub1, sub2
img_adata.write_h5ad(plot_dir+"/results/all_features_logged.h5ad")

```

#### 5.3 Understand masks

- num_samples: the number of samples for each mask visualization


```python
# Summarize the properties of each mask
ret=mph.mask_properity(masks, img, patch_info, d0, d1, center=True)
print(ret) 

```
       per_contain  per_area           avg_rgb
    0        1.000     0.374  [156.  52.  90.]
    1        1.000     0.345  [186.  71. 110.]
    2        0.988     0.065  [242. 219. 225.]
    3        1.000     0.124  [213. 116. 155.]
    4        1.000     0.181     [98. 29. 64.]
    5        1.000     0.072  [229. 155. 186.]



```python
# Plot some sample masks for visuallization
num_samples = 3 # the number of samples for each mask
for channel in range(masks.shape[0]):
    ret_img=mph.mask_example(channel, img_adata, patch_info, patches, masks, plot_dir=plot_dir+"/figures", num_samples=num_samples, filter_mask_area=True)
    ret_img_cvt=cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)
    plt.imshow(ret_img_cvt)
    plt.axis('off')
    plt.show()
    plt.close()
    
```

**Mask 0 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Mask 1:<br>**
<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/sample_for_mask_0.png" width=25% height=25%> <img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/sample_for_mask_1.png" width=25% height=25%>

**Mask 2 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Mask 3:<br>**
<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/sample_for_mask_2.png" width=25% height=25%> <img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/sample_for_mask_3.png" width=25% height=25%>

**Mask 4 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Mask 5:<br>**
<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/sample_for_mask_4.png" width=25% height=25%> <img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/sample_for_mask_5.png" width=25% height=25%>

#### 5.4 Find major masks


```python
# Calculate the area proportion of each mask
mask_area_prop={}
for i in range(masks.shape[0]):
    tmp_prop=[]
    for j in range(masks.shape[1]):
        tmp_prop.append(np.round(np.sum(masks[i, j, ...])/(patch_size*patch_size),3))
    print("Mask ", i)
    mask_area_prop["Mask_"+str(i)]=tmp_prop

```

``` python
# Generate box-plots to check mask area proportions
dat_bxplt=pd.DataFrame({
    "Value": [value for values in mask_area_prop.values() for value in values],
    "Group": [group for group, values in mask_area_prop.items() for _ in values]})

sns.boxplot(x="Group", y="Value", data=dat_bxplt, palette="Blues")
plt.title("Area proportion of each mask within patches", fontsize=16)
plt.xlabel("Group", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.savefig(plot_dir+"/figures/mask_area_proportion_boxplot.png", dpi=300)
plt.show()
plt.close()
plt.clf()

```

<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/mask_area_proportion_boxplot.png" width=50% height=50%>

From the box-plots, we can find that Mask 0 and Mask 1 capture the most dominant tissue structures.


### 6. Link image features with gene expression
- If users prefer to skip the image extraction and directly proceed with linkage analysis, the pre-generated image features are made available in the [results folder](https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/results).

#### 6.1 Preprocessing


```python
# Gene expression
gene_adata=sc.read("./toy_data/exp_tumor.h5ad")
gene_adata.X=(np.array(gene_adata.X.A) if issparse(gene_adata.X) else np.array(gene_adata.X))
sc.pp.log1p(gene_adata)

# Histology image
img_adata=sc.read(plot_dir+"/results/all_features_logged.h5ad")
img_adata.X=(img_adata.X.A if issparse(img_adata.X) else img_adata.X)
img_adata=img_adata[img_adata.obs.index.isin(gene_adata.obs.index)]
# Keep image features with over 10% non median 
img_adata=img_adata[:, np.sum(img_adata.X!=np.median(img_adata.X, 0), 0)>(img_adata.shape[0]/10)]

```

#### 6.2 Spatial clustering on gene expression and image features separately

Apart from louvain clustering, other spatial clustering methods (e.g., SpaGCN) can also be employed


```python
# Set colors
cnt_color = clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370',"#EAE7CC","#FD1593"], N=256)
cat_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
```

```python
# Gene expression
# Louvain clustering
pca = PCA(n_components=50)
pca.fit(gene_adata.X)
embed=pca.transform(gene_adata.X)
tmp=sc.AnnData(embed)
sc.pp.neighbors(tmp, n_neighbors=10)
sc.tl.leiden(tmp,resolution=0.1)
y_pred=tmp.obs['leiden'].astype(int).to_numpy()
gene_adata.obs["gene_pred"]=y_pred
# or by SpaGCN
gene_adata.obs["gene_pred"]=gene_adata.obs["spagcn_pred"].astype('category') # use the spatial clustering results from SpaGCN
```

```python
# check spatial clustering of gene expression
domains="gene_pred"
num_domains=len(gene_adata.obs[domains].unique())
gene_adata.uns[domains+"_colors"]=list(cat_color[:num_domains])
ax=sc.pl.scatter(gene_adata,alpha=1,x="pixel_y",y="pixel_x",color=domains,title=domains,color_map=cat_color,show=False,size=150000/gene_adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig(plot_dir+"/figures/gene_pred.png", dpi=300)
plt.show()
plt.close()
plt.clf()
# ax=spg.plot_spatial_domains_ez_mode(gene_adata, domain_name="gene_pred", x_name="pixel_y", y_name="pixel_x", plot_color=cat_color, size=150000/gene_adata.shape[0], 
	# show=False, save=True,save_dir=plot_dir+"/figures/gene_pred.png")

```

<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/gene_pred.png" width=75% height=75%>


```python
# Image features
# Louvain clustering
pca = PCA(n_components=50)
pca.fit(img_adata.X)
embed=pca.transform(img_adata.X)
tmp=sc.AnnData(embed)
sc.pp.neighbors(tmp, n_neighbors=10)
sc.tl.leiden(tmp,resolution=0.05)
y_pred=tmp.obs['leiden'].astype(int).to_numpy()
len(np.unique(y_pred)) # number of louvain clusters for image features
img_adata.obs["img_pred"]=y_pred
img_adata.obs["img_pred"]=img_adata.obs["img_pred"].astype('category')
```

```python
# check spatial clustering of image features
domains="img_pred"
num_domains=len(img_adata.obs[domains].unique())
img_adata.uns[domains+"_colors"]=list(cat_color[:num_domains])
ax=sc.pl.scatter(img_adata,alpha=1,x="pixel_y",y="pixel_x",color=domains,title=domains,color_map=cat_color,show=False,size=150000/img_adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig(plot_dir+"/figures/img_pred.png", dpi=300)
plt.show()
plt.close()
plt.clf()
# ax=spg.plot_spatial_domains_ez_mode(img_adata, domain_name="img_pred", x_name="pixel_y", y_name="pixel_x", plot_color=cat_color,size=180000/img_adata.shape[0], 
	# show=False, save=True,save_dir=plot_dir+"/figures/img_pred.png")

```

<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/img_pred.png" width=75% height=75%>    


#### 6.3 Identify subregions


```python
# check spatial clustering of combined clusters
gene_clusters=gene_adata.obs["gene_pred"].tolist()
img_clusters=img_adata.obs["img_pred"].tolist()

# for any cluster pair if the overlapping spots / overall spots > max_threshod (default value is 0.2) then merge the two clusters
gene_adata.obs["gene_img_pred"]=mph.combine_clusters(gene_clusters, img_clusters, min_threshold=1/5, max_threshold=1/2)
gene_adata.obs["combined_pred"]=gene_adata.obs["combined_pred"].astype('category')
# ax=spg.plot_spatial_domains_ez_mode(gene_adata, domain_name="combined_pred", x_name="pixel_y", y_name="pixel_x", plot_color=cat_color,size=150000/gene_adata.shape[0], 
	# show=False, save=True,save_dir=plot_dir+"/figures/combined.png")

```

```python
# Plot subregion
domains="combined_pred"
num_domains=len(gene_adata.obs[domains].unique())
gene_adata.uns[domains+"_colors"]=list(cat_color[:num_domains])
ax=sc.pl.scatter(gene_adata,alpha=1,x="pixel_y",y="pixel_x",color=domains,title=domains,color_map=cat_color,show=False,size=150000/img_adata.shape[0])
ax.set_aspect('equal', 'box')
ax.axes.invert_yaxis()
plt.savefig(plot_dir+"/figures/combined_pred.png", dpi=300)
plt.show()
plt.close()
plt.clf()

```
<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/combined_pred.png" width=75% height=75%>

#### 6.4 Quantify the curve-based similarity

- genes: a set of interested genes or identified from DE analysis
- channel: the mask channel number to focus on
- w_cor: the weights for correlation (default value is 0.5)
- CPSI: Curve-based Pattern Similarity Index


```python
# Specify a set of genes (or from DE analysis) that are related to the interested biological process
# e.g., a set of genes related to antigen presentation
genes_set=['HLA-F', 'HHLA3', 'HLA-DR', 'CD1D', 'IFNG', 'LMP7', 'VCAM1', 'RFXANK', 'ERAP2', 'CD274', 'PDCD1', 'LMP2', 'TAPBPL', 'HLA-DQ', 'HLA-DP', 'ERAP1', 'HLA-DMA', 'CD40', 'IDO1', 'IFI16', 'HLA-E', 'HLA-DMB', 'RFX5', 'AP1M1', 'TAP2', 'TAP1', 'HHLA2', 'LMP10', 'CD80', 'PSMB8', 'CALR', 'CD74', 'HHLA1', 'RFXAP', 'CD86', 'CD70', 'CIITA', 'CTLA4', 'TAPBP', 'PSMB10', 'MR1', 'PSMB9', 'NLRC5', 'HLA-G', 'ICOS', 'CD40LG', 'SEC61', 'IRF1', 'CD276', 'ICAM1', 'B2M']
filtered_genes_set=list(set(genes_set) & set(gene_adata.var.index.tolist()))

# Calculate the spatial similarity between generated image features and selected genes set by CPSI
channel=4 # specify the target mask channel
CPSI=mph.cpsi_eva(gene_adata, img_adata, filtered_genes_set, channel)

```


#### 6.5 Generate marginal curves


```python
# e.g., gene CD74
g="CD74"

# Identify the image feature that has the highest CPSI with the target gene and generate a gradient marginal curve along x-axis and y-axis for the target pair of gene expression and image feature
range_step=1/4
num_cuts=5
f, x, y, _=mph.marginal_curve(gene_adata, img_adata, CPSI, g, range_step, num_cuts)

```
The identified image feature having the highest CPSI with the target gene CD74: c4_solidity_iqr


```python
# Generate a scatter plot for x and y
plt.scatter(x, y, s=80, c='blue', alpha=0.75)
plt.xlabel("gene expression levels", fontsize=14)
plt.ylabel("image feature levels", fontsize=14)
plt.title("The regional linkage between "+g+" and "+f, fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
plt.close()
plt.clf()

```

<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/scatterplot_CD74_c4_solidity_iqr.png" width=50% height=50%>    

#### 6.6 Statistical test to evaluate the confidence of the selected image feature

```python
# perform a two-sample one-sided t-test
target_f_scores=CPSI.loc[:,f].values.flatten()
other_f_scores=CPSI.loc[:,CPSI.columns!=f].values.flatten()
scipy.stats.ttest_ind(target_f_scores, other_f_scores,alternative="greater")

```
TtestResult(statistic=13.484389257247123, pvalue=5.849801060163686e-41, df=4520.0)

p-value is smaller than 0.05, indicating that the selected image feature has significantly higher CPSIs with the target set of genes compared to other image features.

```python
# Generate a histogram to check CPSIs distribution
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

```

<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/CPSIs_distribution.png" width=65% height=65%>


### 7. Select samples for visual demonstration

- num_sample: the number of samples for demonstrating the linkage between the pair of gene expression feature and image feature


```python
labels=np.load(plot_dir+"/results/cc_no_details.npy")
```


```python
# Load in the generated patch_info, patches, and labels
# plot_dir="."
# patch_info=pd.read_csv(plot_dir+"/results/patch_info.csv", header=0, index_col=0)
# patches=np.load(plot_dir+"/results/patches.npy")
# labels=np.load(plot_dir+"/results/cc_no_details.npy")

# Specify a set of interested image features
target_features = [f]
visual_img_list = []
num_sample=5

for f in target_features:
	if not os.path.exists(plot_dir+"/figures/"+f):
		os.mkdir(plot_dir+"/figures/"+f)
	visual_img=mph.sample_illustration(f, img_adata, patch_info, patches, labels, plot_dir=plot_dir+"/figures/"+f, num_cuts=num_cuts, range_step=range_step, num_sample=num_sample, filter_mask_area=True, filter_cc_q100=False)
	visual_img_list.append(visual_img)


```


```python
# Read in arrow image 
arrow_img=cv2.imread(plot_dir+"/figures/arrow.png", cv2.IMREAD_UNCHANGED)
arrow_img=cv2.cvtColor(arrow_img, cv2.COLOR_BGR2RGB) # convert BGR to RGB (already a numpy array)

# Generate sample linkage visual demonstration
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
    
```

<img src="https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures/linkage_demonstration_c4_solidity_iqr+arrow.png" width=100% height=100%>


### 8. Parameter settings
**Patch segmentation:** $k$, $t$, $\alpha$.
- $k$: the number of initial clusters (default value is 10 and recommend using the default).
- $t$: the threshold used to control the integrity of clusters for spatial smoothing (default value is 4 and recommend using the default).
- $\alpha$: the threshold used to control color distances in cluster merging (default value is 30, and it can be set to 20 for thinner structures).

**Mask matching:** $\alpha$.
- $\alpha$: same as above.

**Subregion devision:** $\beta$.
- $\beta$: Jaccard index to evaluate the overlapping between cluster pairs (default value is 0.2 and recommend using the default).

**Calculating marginal curves:** $l$.
- $l$: the parameter of window size used to control the number of intervals within a subregion (default value is 100, and it can be set to 50 for capturing coarse marginal patterns).

