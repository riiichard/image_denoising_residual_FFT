# Image Denoising using Deep Residual Blocks with Fourier Transform

## Introduction

Image Denoising is critical in image processing pipeline and computer vision tasks. Existing developed methods are mainly traditional spatial filtering algorithms and real-valued deep learning methods. 

The spectrums of the clean and noisy images are quite different.

![Alt text](/code/intro.png "Table 1. Summary of Some Representative Models Trained")

This study aims to propose an innovative neural network model that filters out noise in **both the time and frequency domain**, along with the use of **residual blocks**.

## Setup

Requires **Python3.10**. Required libraries are listed in the corresponding files.

Run **BM3D.py** in BM3D directory to get denoising results using BM3D.

File **models.py** implements the neural networks models used in this study.

Run **train.py** and **test.py** to train and evaluate the models respectively. You might specify (or use default values) and tune the following parameters to get better performances: **batchSize**, **cnn_model** which is one of dncnn, rescnn, and fftrescnn, **num_of_layer**, **num_of_resblocks**, and **lr**.

## Dataset

Blind Training: BSDS400 (#=20,000), Testing: SET12 & BSD68

## Proposed model architecture (ResDnCNN, FFTResCNN):

![Alt text](/code/model.png "Figure 1. Frequency of each response variable level for each year")

## Some preliminary results

![Alt text](/code/results.png "response variable level for each year")

## Acknowledgement

Collaborated with three groupmates: Xiaoran(Sharon) Liu and Wenzhe Xu.
