# Multi-label-Classification-of-Blood-Cells-Using-CNN
## Requirements
* PyCharm
* PyTorch
* Python
* Convolutional Neural Network (CNN)

## Introduction
This project used PyTorch to build a CNN model to recognize all types of cells that are present in the given images. These cell types are: red blood cell, difficult, gametocyte, trophozoite, ring, schizont, and leukocyte. The best model over the past seven-day data challenge competition yielded a loss score of 4.54085.

## Dataset
The training set contains 929 rectangular cell images of varied size, 929 txt files with the corresponding string labels, and 929 json files with the cell detection bounding boxes. During the course of the 7-day competition, only the raw images and the txt files were used to train the model. Since the dataset is unbalanced and the raw data (.png and .txt) cannot be directly used to train the network, the following steps were performed to preprocess the data.

## Data Preprocessing
* Image Resizing
* Data Augmentation
* Labeling
* Loading Data using DataLoader
Detailed description is provided in the project report under the **Report** folder.

## Modeling
In this project, I built a network that contains 12 layers.5 As illustrated in Figure 1, the model follows the block design pattern, where one convolutional layer (filters of size 3 X 3), one max pooling layer (factor 2), and one ReLU activation layer are stacked as a single learning block. The whole model is formed by four learning blocks, followed by three fully connected layer, and lastly following by a sigmoid activation function to render the probabilities of each class independently for classification. I chose Adam as the model optimizer and BCELoss as the function loss.
![alt text](https://github.com/weining20000/Multi-Label-Classification-of-Blood-Cells-Using-CNN/blob/master/Report/figure_1.png)

