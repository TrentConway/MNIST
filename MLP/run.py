#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as torch
import torchvision as torchvision
import matplotlib.pyplot as plt
from MLP import Net 
from plot import plot
from dataLoader import DataLoader
from parameters import Parameters



# initialise model variables 
modelParameters = Parameters('parameters.yml').getModelParams()
print(modelParameters)

# initialise dataloaders
dataLoader = DataLoader()
trainingLoader = dataLoader.trainingData()
testLoader = dataLoader.testData()

# Visualise the data
examples = enumerate(testLoader)
batch_idx, (example_data, example_target) = next(examples)

plot(example_data, example_target, "MNIST dataset")


# Run the model
torch.backends.cudnn.enabled = False
torch.manual_seed(modelParameters['random-seed'])

network = Net()
optimizer = torch.optim.SGD(network.parameters(), lr = modelParameters['learning-rate'], momentum = modelParameters['momentum']) 

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainingLoader.dataset) for i in range(modelParameters['n-epochs'] + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(trainingLoader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % modelParameters['log-interval'] == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(trainingLoader.dataset),
        100. * batch_idx / len(trainingLoader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(trainingLoader.dataset)))

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in testLoader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(testLoader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(testLoader.dataset),
    100. * correct / len(testLoader.dataset)))

test()
for epoch in range(1, modelParameters['n-epochs'] + 1):
  train(epoch)
  test()


# Visualise the Results
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right') 
plt.xlabel('number of training examples seen')

plt.ylabel('negative log likelihood loss')
plt.show()

with torch.no_grad():
    output = network(example_data)
    example_predictions = [output.data.max(1, keepdim=True)[1][i].item() for i in range(modelParameters['batch-size'])] 

plot(example_data, example_predictions, "MLP Predictions")
