# Predictive principal component analysis (PredPCA)

[![DOI](https://zenodo.org/badge/245048386.svg)](https://zenodo.org/badge/latestdoi/245048386)

## Overview
Predictive principal component analysis (PredPCA) is an analytically solvable unsupervised learning scheme that extracts the most informative components for predicting future inputs. It is a convex optimization and can find the analytical expressions of optimal weight matrices and encoding dimensionality that provide the global minimum of the squared test prediction error.

This project involves MATLAB scripts for PredPCA

<br>

This is a supplementary material of

"Dimensionality reduction to maximize prediction generalization capability"

Takuya Isomura, Taro Toyoizumi

Nature Machine Intelligence 3, 434–446 (2021). https://doi.org/10.1038/s42256-021-00306-1

Preprint version is available at https://arxiv.org/abs/2003.00470

<br>

Copyright (C) 2020 Takuya Isomura

(RIKEN Center for Brain Science)

<br>

2020-12-18


## System Requirements
This package requires only a standard computer with enough RAM to support the in-memory operations.

Software: MATLAB

RAM: 16+ GB for fig2 and fig3 / 128+ GB for fig4

<br>

The package has been tested on the following system:

iMac Pro (macOS Mojave)

CPU: 2.3GHz 18 core Intel Xeon W

RAM: 128 GB

MATLAB R2019b

The runtimes below are generated using this setup.


## Demo
### fig2
Download all scripts under the 'fig2' directory and put them in the same directory.

Before run this script, please download MNIST dataset from http://yann.lecun.com/exdb/mnist/ and expand 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', and 't10k-labels-idx1-ubyte' in the same directory.

Run 'fig2a.m', 'fig2b.m', and 'fig2c.m'.

Each script should take less than 2 minutes. Examples of their outcomes are stored in 'fig2/output'.

<br>

In order to evaluate the performance of autoencoder, please download the scripts from http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html and train weight matrices with digit images involving monochrome inversion. Put an output file 'mnist_weights.mat' in the same directory before running 'fig2b.m'.

### fig3
Download all scripts under the 'fig3' directory and put them in the same directory.

Before run this script, please download ALOI dataset from http://aloi.science.uva.nl (full color (24 bit), quarter resolution (192 x 144), viewing direction) and expand 'aloi_red4_view.tar' in the same directory.

Reference to ALOI dataset: Geusebroek JM, Burghouts GJ, Smeulders AWM, The Amsterdam library of object images, Int J Comput Vision, 61, 103-112 (2005)

Run 'predpca_aloi_preprocess.m' and put the output file 'aloi_data.mat' in the same directory.

Run 'fig3_predpca.m' to perform PredPCA and optain the outcomes.

'predpca_aloi_preprocess.m' should take approximately 10 minutes. 'fig3_predpca.m' should take approximately 30 minutes. Examples of the outcomes are stored in 'fig3/output'.

### fig4
Download all scripts under the 'fig4' directory and put them in the same directory.

Before run this script, please download BDD100K dataset from https://bdd-data.berkeley.edu, downsample the videos to 320 x 180 and put them in the same directory.

'predpca_bdd100k_preprocess.m' generates compressed input data fed to PredPCA.

'predpca_bdd100k_training.m' runs PredPCA to learn the optimal weight matrices.

'predpca_bdd100k_test.m' predicts 0.5 second future of test data.

'predpca_bdd_cat.m' and 'predpca_bdd_dyn.m' extract categorical and dynamical features, respectively.

The preprocessing should take 10+ hours.


## License
This project is covered under the **GNU General Public License v3.0**.
