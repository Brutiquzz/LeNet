import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

class LeNet(nn.Module):

    def __init__(self):
      super(LeNet, self).__init__()
      self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1)
      self.subSampl2 = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv3 = nn.Conv2d(6, 16, kernel_size=3, stride=1)
      self.subSampl4 = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv5 = nn.Conv2d(16, 120, kernel_size=3, stride=1)
      self.fc6 = nn.Linear(1080, 10)

      self.EpochRunner = True
      self.CurrentValidationLoss = []
      self.OverallValidationLoss = []
      self.BetsValidationLoss = sys.maxsize


    def forward(self, input):
        output = torch.sigmoid(self.conv1(input)) # 26
        output = self.subSampl2(output) # 13
        output = torch.sigmoid(self.conv3(output)) # 11
        output = self.subSampl4(output) # 5-6
        output = torch.sigmoid(self.conv5(output))
        output = output.view(-1, self.num_flat_features(output))
        output = self.fc6(output)
        return output

    def predict(self, input):
        input = self.forward(input)
        x = input.detach().numpy()
        return np.argmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

    def EvaluateNextEpoch(self):
        data = np.array(self.OverallValidationLoss)

        potentialMin = np.min(data)
        if potentialMin < self.BetsValidationLoss :
            self.BetsValidationLoss = potentialMin
            torch.save(self.state_dict(), "bestState.pth")
        elif len(data) >= 20:
            if not (np.isin(self.BetsValidationLoss,data[(len(data)-30):])):
              self.EpochRunner = False