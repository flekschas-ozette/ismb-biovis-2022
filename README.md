# Code for our BioVis Talk at ISMB '22

This repository contains the code to reproduce all plots presented in our
BioVis talk at ISMB '22.

For a more elaborate R implementation that includes our clustering method, please take a look at https://github.com/RGLab/FAUST.

For details about the method, please take a look at the related publication:

[Greene et al., 2021, New interpretable machine-learning method for single-cell data reveals correlates of clinical response to cancer immunotherapy. _Pattern_.](https://www.sciencedirect.com/science/article/pii/S2666389921002348)

## Requirements

- [Conda](https://github.com/conda-forge/miniforge)

## Install

First, get the code and install the conda environment.

```
git clone git@github.com:flekschas-ozette/ismb-biovis-2022.git
cd ismb-biovis-2022
conda env create -f environment.yml
conda activate ozette-ismb-biovis-2022
```

Next, download the example data from https://figshare.com/articles/dataset/ISMB_BioVis_2022_Data/20301639 and place it under `data/mair-2022`.

## Get Started

1. Start JupyterLab:

   ```
   jupyter-lab
   ```

2. Open one of the following notebooks:

  - Explanation of our transformation embedding approach: [annotation-embedding.ipynb](http://localhost:8888/lab/tree/annotation-embedding.ipynb)
    
  - Comparison of our transformation approach using different non-linear embedding methods: [compare-annotation-embedding.ipynb](http://localhost:8888/lab/tree/compare-annotation-embedding.ipynb)
    
  - Joint embedding of two samples showing how our transformation approach helps to reduce batch effects: [joint-annotation-embedding.ipynb](http://localhost:8888/lab/tree/joint-annotation-embedding.ipynb)
