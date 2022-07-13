# Code for our BioVis Talk at ISMB '22

This repository contains the code to reproduce all plots presented in our
BioVis talk at ISMB '22.

## Requirements

- [Conda](https://github.com/conda-forge/miniforge)

## Install

```
git clone git@github.com:flekschas-ozette/ismb-biovis-2022.git
cd ismb-biovis-2022
conda env create -f environment.yml
conda activate ozette-ismb-biovis-2022
```

## Get Started

1. Start JupyterLab:

   ```
   jupyter-lab
   ```

2. Open one of the following notebooks:

  - Explanation of our transformation embedding approach: [annotation-embedding.ipynb](http://localhost:8888/lab/tree/annotation-embedding.ipynb)
    
  - Comparison of our transformation approach using different non-linear embedding methods: [compare-annotation-embedding.ipynb](http://localhost:8888/lab/tree/compare-annotation-embedding.ipynb)
    
  - Joint embedding of two samples showing how our transformation approach helps to reduce batch effects: [joint-annotation-embedding.ipynb](http://localhost:8888/lab/tree/joint-annotation-embedding.ipynb)
