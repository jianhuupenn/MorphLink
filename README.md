# MorphLink  v1.0.1

## MorphLink: Bridging Cell Morphological Behaviors and Molecular Dynamics in Multi-modal Spatial Omics


#### Jing Huang, Chenyang Yuan, Jiahui Jiang, Jianfeng Chen, Sunil S. Badve, Yesim Gokmen-Polar, Rossana L. Segura, Xinmiao Yan, Alexander Lazar, Jianjun Gao, Michael Epstein, Linghua Wang* and Jian Hu*

MorphLink is a computational framework to systematically extract and link interpretable morphological features with molecular measurements in multi-model spatial omics analyses. The identified linkages provide a transparent depiction of cellular behaviors that drive transcriptomic heterogeneity and immune diversity across different regions within diseased tissues. Moreover, MorphLink is scalable and robust against cross-sample batch effects, making it an efficient method for integrating spatial omics data analysis across samples, cohorts, and modalities, and enhancing the interpretation of results for large-scale studies. MorphLink is applicable to various types of spatial omics data, including spatial transcriptomics (Spatial Transcriptomics, 10x Visium, 10x Xenium, and MERFISH), spatial proteomics (CODEX and IMS), and the simultaneous measurement of proteins and transcriptome (spatial CITE-seq and CosMx). 

![MorphLink workflow](docs/asserts/images/workflow.jpg)
<br>
For thorough details, see the preprint: [Biorxiv](https://www.biorxiv.org/)
<br>

## Usage

With [**MorphLink**](https://github.com/jianhuupenn/MorphLink) package, you can:

- Extract interpretable morphological features from histology images in a label-free manner.
- Quantify the relationships between cell morphological and molecular features in a spatial context.
- Visually examine how cellular behavior changes from both morphological and molecular perspectives.


## Tutorial

For the step-by-step tutorial, please refer to: 
<br>
https://github.com/jianhuupenn/
<br>
A Jupyter Notebook of the tutorial is accessible from : 
<br>
https://github.com/jianhuupenn/
<br>
Toy data and results can be downloaded at: 
<br>
https://drive.google.com/drive/folders/
<br>
Please install Jupyter in order to open this notebook.

## System Requirements
Python support packages: igraph, torch, pandas, numpy, scipy, scanpy > 1.5, anndata, louvain, sklearn.

## Versions the software has been tested on
Environment 1:
- System: Mac OS 10.13.6
- Python: 3.7.0
- Python packages: pandas = 1.1.3, numpy = 1.18.1,numba=0.53.1 python-igraph=0.7.1,torch=1.5.1,louvain=0.6.1,scipy = 1.4.1, scanpy = 1.5.1, anndata = 0.6.22.post1, natsort = 7.0.1, sklearn = 0.22.1

Environment 2:
- System: Anaconda
- Python: 3.7.9
- Python packages: pandas = 1.1.3, numpy = 1.20.2, python-igraph=0.8.3, torch=1.6.0,louvain=0.7.0, scipy = 1.5.2, scanpy = 1.6.0, anndata = 0.7.4, natsort = 7.0.1, sklearn = 0.23.3

Environment 3:
- System: Anaconda
- Python: 3.8.8
- Python packages: pandas = 1.2.4, numpy = 1.19.1, python-igraph=0.9.1, torch=1.8.1, louvain=0.7.0, scipy = 1.6.3, scanpy = 1.7.2, anndata = 0.7.6, natsort = 7.1.1, sklearn = 0.24.2


## Contributing

Souce code: [Github](https://github.com/jianhuupenn/MorphLink)  

We are continuing adding new features. Bug reports or feature requests are welcome. 

Last update: 06/23/2024, version 1.0.1



## References

Please consider citing the following reference:

- https://www.

<br>
