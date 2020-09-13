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

# Create MLP network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1= torch.nn.Linear(in_features = 784, out_features = 256) 
        self.fc2= torch.nn.Linear(in_features = 256, out_features = 64) 
        self.fc3= torch.nn.Linear(in_features = 64, out_features = 10) 
        self.relu = torch.nn.ReLU()        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x) 
        x = self.relu(x)  
        x = self.fc2(x) 
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

n_epochs = 16
batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.0001
momentum = 0.2
log_interval = 100

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

network = Net()
optimizer = torch.optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum) 

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainingLoader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(trainingLoader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
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
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

with torch.no_grad():
  output = network(example_data)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
plt.show()

