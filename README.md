# MorphLink  v1.0.7

## Bridging cell morphological behaviors and molecular dynamics in multi-modal spatial omics with MorphLink


#### Jing Huang, Chenyang Yuan, Jiahui Jiang, Jianfeng Chen, Sunil S. Badve, Yesim Gokmen-Polar, Rossana L. Segura, Xinmiao Yan, Alexander Lazar, Jianjun Gao, Bing Yao, Michael Epstein, Linghua Wang* and Jian Hu*

MorphLink is a computational framework to systematically extract and link interpretable morphological features with molecular measurements in multi-model spatial omics analyses. The identified linkages provide a transparent depiction of cellular behaviors that drive transcriptomic heterogeneity and immune diversity across different regions within diseased tissues. Moreover, MorphLink is scalable and robust against cross-sample batch effects, making it an efficient method for integrating spatial omics data analysis across samples, cohorts, and modalities, and enhancing the interpretation of results for large-scale studies. MorphLink is applicable to various types of spatial omics data, including spatial transcriptomics (Spatial Transcriptomics, 10x Visium, 10x Xenium, and MERFISH), spatial proteomics (CODEX and IMS), and the simultaneous measurement of proteins and transcriptome (spatial CITE-seq and CosMx). 

![MorphLink workflow](docs/asserts/images/workflow.jpg)
<br>
For thorough details, see the preprint: [Biorxiv](https://www.biorxiv.org/content/10.1101/2024.08.24.609528v1)
<br>

## Usage

With [**MorphLink**](https://github.com/jianhuupenn/MorphLink) package, you can:

- Extract interpretable morphological features from histology images in a label-free manner.
- Quantify the relationships between cell morphological and molecular features in a spatial context.
- Visually examine how cellular behavior changes from both morphological and molecular perspectives.


## Tutorial

For the step-by-step tutorial, please refer to: 
<br>
https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/tutorial.md
<br>
<br>
A Jupyter Notebook of the tutorial is accessible from: 
<br>
https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/tutorial.ipynb
<br>
Please install Jupyter in order to open this notebook.
<br>
<br>
Toy data can be downloaded at: 
<br>
https://drive.google.com/drive/folders/1NgJICg1jFD2HP7WGZ9vXk7GrRJRoFfSD?usp=sharing
<br>
Pre-generated image features can downloaded at:
<br>
https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/results
<br>
Intermediate results and figures are saved at:
<br>
https://github.com/jianhuupenn/MorphLink/blob/main/tutorial/figures
<br>
Intermediate patches and masks are saved at:
<br>
https://drive.google.com/drive/folders/1fzeyG0rbpl83AL0Qu15Pu0yDg48cJXxL?usp=sharing
<br>
<br>

## System Requirements
Python support packages: pandas, numpy, numba, scipy, scanpy, anndata, scikit-learn, scikit-image, matplotlib, imutils, opencv-python, leidenalg.

## Versions the software has been tested on
Environment 1:
- System: Mac OS Sonoma 14.0 (M1 Pro)
- Python: 3.11.5
- Python packages: pandas = 2.1.4, numpy = 1.26.2, numba = 0.58.1, scipy = 1.11.4, scanpy = 1.9.6, anndata = 0.10.3, scikit-learn = 1.3.2, scikit-image = 0.23.2, matplotlib = 3.8.2, imutils = 0.5.4, opencv-python = 4.8.1, leidenalg = 0.10.2.

Environment 2:
- System: Anaconda (23.9.0)
- Python: 3.11.5
- Python packages: pandas = 2.0.3, numpy = 1.24.3, numba = 0.57.1, scipy = 1.11.1, scanpy = 1.9.8, anndata = 0.10.8, scikit-learn = 1.3.0, scikit-image = 0.20.0, matplotlib = 3.7.2, imutils = 0.5.4, opencv-python = 4.10.0, leidenalg = 0.10.2.


## Contributing

Souce code: [Github](https://github.com/jianhuupenn/MorphLink)  

We are continuing adding new features. Bug reports or feature requests are welcome. 

Last update: 10/28/2025, version 1.0.7. (updated environment setup instructions in the tutorial)


### New in v1.0.7:
- Added new packaged functions for common tasks into **tutorial_util.py** for easier use and improved readability.
- Now users can run the full pipeline with only a few function calls.
- Added new helper functions:
  - `test_patch_size`: determine the suitable image patch sizes for different spatial transcriptomics techniques data.
  - `cpsi_eva` and `cpsi_distri_histo`: evaluate spatial linkages between the target image feature and a set of interested genes, with histogram-based visualization of distribution differences.
  - `sample_linkage_visualization`: generate visual demonstration of the identified gene-morphology linkage.

## References

Please consider citing the following reference:

[Huang, J., Yuan, C., Jiang, J., Chen, J., Badve, S. S., Gokmen-Polar, Y., ... & Hu, J. (2025). Bridging cell morphological behaviors and molecular dynamics in multi-modal spatial omics with MorphLink. Nature Communications, 16(1), 5878.](https://doi.org/10.1038/s41467-025-61142-0)  

<br>
