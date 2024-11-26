import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MorphLink", 
    version="1.0.4",
    author="Jing Huang",
    author_email="jing.huang@emory.edu",
    description="MorphLink: Bridging Cell Morphological Behaviors and Molecular Dynamics in Multi-modal Spatial Omics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jianhuupenn/MorphLink",
    packages=setuptools.find_packages(),
    install_requires=["numpy","pandas","numba","anndata","scipy","scanpy","scikit-learn","scikit-image","matplotlib","slideio","imutils","opencv-python","leidenalg"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)