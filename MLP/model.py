import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Model():
    def __init__(self, network, optimizer, dataLoader, modelParameters):
        self.network = network
        self.optimizer = optimizer
        self.trainingLoader = dataLoader.trainingData()
        self.testLoader = dataLoader.testData()
        self.modelParameters = modelParameters

        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.modelParameters['random-seed'])

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = [i*len(self.trainingLoader.dataset) for i in range(self.modelParameters['n-epochs'] + 1)]

    def train(self, epoch):
        self.network.train()
        for batch_idx, (data, target) in enumerate(self.trainingLoader):
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.modelParameters['log-interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), 
                    len(self.trainingLoader.dataset),
                    100. * batch_idx / len(self.trainingLoader), 
                    loss.item())
                )
                self.train_losses.append(loss.item())
                self.train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(self.trainingLoader.dataset)))

    def test(self):
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.testLoader:
                output = self.network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.testLoader.dataset)
        self.test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, 
            correct, 
            len(self.testLoader.dataset),
            100. * correct / len(self.testLoader.dataset))
        )

