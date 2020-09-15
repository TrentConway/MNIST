#!/usr/bin/env python3

import torchvision as torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils as utils


class DataLoader():
    def __init__(self):
        self.batchsize  = 32 

    def transformation(self):
        return transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def trainingData(self):
        trainingData = datasets.MNIST(
            "../MNIST/TrainingData", 
            train=True, 
            transform=self.transformation(), 
            download=True)
        trainingLoader = utils.data.DataLoader(
            trainingData, 
            batch_size=self.batchsize, 
            shuffle=True)
        return trainingLoader

    def testData(self):
        testData = datasets.MNIST(
            "../MNIST/TestData", 
            train=False, 
            transform=self.transformation(), 
            download=True)
        testLoader = utils.data.DataLoader(
            testData, 
            batch_size=self.batchsize, 
            shuffle=True)
        return testLoader 
