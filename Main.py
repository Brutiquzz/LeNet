import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from LeNet import LeNet
import torch
import torch.nn as nn
import torch.nn.functional as F

dataTrain = np.array(pd.read_csv("train.csv"))
dataTest = 0
#print(data.shape)

trainLabel = dataTrain[:,0]
trainingSet = dataTrain[:,1:]
train = []
for row in trainingSet:
    train.append(row.reshape(28,28))
train = torch.from_numpy(np.array(train))

#print(trainLabel[240])
#plt.imshow(train[240,:, :])
#plt.show()

model = LeNet()
passe = torch.from_numpy(np.expand_dims(np.expand_dims(train[240,:, :], axis=0), axis=0))
passe = passe.type('torch.FloatTensor')
#print(passe.shape)
print(model.forward(passe))
