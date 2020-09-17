#!/usr/bin/env python3

import torch as torch
import torch.optim as optim
import matplotlib.pyplot as plt
from MLP import Net 
from plot import plot
from dataLoader import DataLoader
from parameters import Parameters
from model import Model

# initialise model variables 
modelParameters = Parameters('parameters.yml').getModelParams()

# initialise dataloaders
dataLoader = DataLoader()
trainingLoader = dataLoader.trainingData()
testLoader = dataLoader.testData()

# Visualise the data
examples = enumerate(testLoader)
batch_idx, (example_data, example_target) = next(examples)
plot(example_data, example_target, "MNIST dataset")


# Initialise the network & model
network = Net()
print(network)
optimizer = torch.optim.SGD(network.parameters(), lr = modelParameters['learning-rate'], momentum = modelParameters['momentum']) 
model = Model(network, optimizer, dataLoader, modelParameters)

# train the network
print("BASE LINE -- RANDOM WEIGHTS")
model.test()
for epoch in range(1, modelParameters['n-epochs'] + 1):
    print("===================================")
    print("\n EPOCH: {} \n".format(epoch))
    model.train(epoch)
    model.test()


# Visualise the Learning Rate
fig = plt.figure()
plt.plot(model.train_counter, model.train_losses, color='blue')
plt.scatter(model.test_counter, model.test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right') 
plt.xlabel('number of training examples seen')

plt.ylabel('negative log likelihood loss')
plt.show()

# Visualise the models predictions
with torch.no_grad():
    output = model.network(example_data)
    example_predictions = [output.data.max(1, keepdim=True)[1][i].item() for i in range(modelParameters['batch-size'])] 

plot(example_data, example_predictions, "MLP Predictions")
