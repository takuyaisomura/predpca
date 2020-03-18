# Predictive principal component analysis (PredPCA)

## Overview
Predictive principal component analysis (PredPCA) is an analytically solvable unsupervised learning scheme that extracts the most informative components for predicting future inputs. It is a convex optimization and can find the analytical expressions of optimal weight matrices and encoding dimensionality that provide the global minimum of the squared test prediction error.

This project involves MATLAB scripts for predictive principal component analysis (PredPCA)

<br>

This is a supplementary material of

"Dimensionality reduction to maximize prediction generalization capability"

Takuya Isomura, Taro Toyoizumi

Preprint at https://arxiv.org/abs/2003.00470

<br>

Copyright (C) 2020 Takuya Isomura

(RIKEN Center for Brain Science)

<br>

2020-3-5


## System Requirements
This package requires only a standard computer with enough RAM to support the in-memory operations.

Software: MATLAB

RAM: 16+ GB for Figs 2,3 / 128+ GB for Fig 4

<br>

The package has been tested on the following system:

iMac Pro (macOS Mojave)

CPU: 2.3GHz 18 core Intel Xeon W

RAM: 128 GB

MATLAB R2019b

The runtimes below are generated using this setup.


## Demos
### For Fig 2
Before run this script, please download MNIST dataset from http://yann.lecun.com/exdb/mnist/ and expand

train-images-idx3-ubyte

train-labels-idx1-ubyte

t10k-images-idx3-ubyte

t10k-labels-idx1-ubyte

in the same directory.

Download all scripts under the 'fig2' directory and put them in the same directory.

Run 'fig2a.m', 'fig2b.m', and 'fig2c.m'.

Each script should take less than 5 minutes.

### For Fig 3

### For Fig 4


## License
This project is covered under the **GNU General Public License v3.0**.
