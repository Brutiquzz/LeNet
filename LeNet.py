import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self):
      super(LeNet, self).__init__()
#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
#torch.nn.Linear(in_features, out_features, bias=True)
      self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1)
      self.subSampl1 = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv3 = nn.Conv2d(6, 16, kernel_size=3, stride=1)
      self.subSampl4 = nn.AvgPool2d(kernel_size=2, stride=2)
      self.conv5 = nn.Conv2d(16, 120, kernel_size=3, stride=1)
      self.fullyLayer = nn.Linear(120, 10)


    def forward(self, input, printDimensions=False):
        print(input.shape)
        output = torch.sigmoid(self.conv1(input)) # 26
        print(output.shape)
        output = self.subSampl1(output) # 13
        print(output.shape)
        output = torch.sigmoid(self.conv3(output)) # 11
        print(output.shape)
        output = self.subSampl4(output) # 5-6
        print(output.shape)
        output = torch.sigmoid(self.conv5(output))
        print(output.shape)
        output = self.fullyLayer(output)
