#!/usr/bin/env python3

import torch as torch
import torchvision as torchvision

# Model Variables 


# Training Loader 
trainingData = torchvision.datasets.MNIST("../MNIST/TrainingData", train=True, transform=None, target_transform=None, download=True)
trainLoader = torch.utils.data.DataLoader(trainingData, batch_size=32, shuffle=True)

# Test Loader
testData = torchvision.datasets.MNIST("../MNIST/TestData", train=False, transform=None, target_transform=None, download=True)
testLoader = torch.utils.data.DataLoader(testData, batch_size=32, shuffle=True)


