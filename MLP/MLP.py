#!/usr/bin/env python3

import torch.nn as nn

# Create MLP network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features = 784, out_features = 64) 
        self.fc2 = nn.Linear(in_features = 64, out_features = 64) 
        self.fc3 = nn.Linear(in_features = 64, out_features = 10)
        self.relu = nn.ReLU()        
        self.logSoftmax= nn.LogSoftmax(dim=1)

    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x) 
        x = self.relu(x)  
        x = self.fc2(x) 
        x = self.relu(x)
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x
