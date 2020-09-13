#!/usr/bin/env python3

import torch as torch
import torchvision as torchvision
import matplotlib.pyplot as plt

# Model Variables 
BATCH_SIZE = 32

# Transform PIL Image to tensor
transformation = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Training Loader 
trainingData = torchvision.datasets.MNIST(
    "../MNIST/TrainingData", 
    train=True, 
    transform=transformation, 
    target_transform=None, 
    download=True)
trainingLoader = torch.utils.data.DataLoader(
    trainingData, 
    batch_size=BATCH_SIZE, 
    shuffle=True)

# Test Loader
testData = torchvision.datasets.MNIST(
    "../MNIST/TestData", 
    train=False, 
    transform=transformation, 
    target_transform=None, 
    download=True)
testLoader = torch.utils.data.DataLoader(
    testData, 
    batch_size=BATCH_SIZE, 
    shuffle=True)


# Visualise the data
examples = enumerate(testLoader)
batch_idx, (example_data, example_target) = next(examples)

fig = plt.figure()
plt.suptitle('Visualise MNIST Test Batch', fontsize=20)
for i in range(BATCH_SIZE):
    plt.subplot(4,8,i+1)
    plt.title("val{}".format(example_target[i]), fontsize=8)
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.axis('off')
plt.show()

