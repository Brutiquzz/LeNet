import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from LeNet import LeNet
import torch
import torch.nn as nn
import torch.nn.functional as F

Training = False
Testing = True
loadState = True

TrainingData = np.array(pd.read_csv("mnist_train.csv"))[0:50000, :]
ValidationData = np.array(pd.read_csv("mnist_train.csv"))[50000:,:]
TestingData = np.array(pd.read_csv("mnist_test.csv"))

model = LeNet()
if loadState:
    model.load_state_dict(torch.load("bestState.pth"))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn =  nn.CrossEntropyLoss()

if Training:
    while model.EpochRunner:
        trainloader = torch.utils.data.DataLoader(TrainingData, batch_size=250, shuffle=True, num_workers=8)
        for batch in trainloader:
            optimizer.zero_grad()

            batchLabel = batch[:,0]
            batchData = batch[:,1:].reshape(250,1,28,28).float()

            y_pred = model.forward(batchData)

            loss = loss_fn(y_pred, batchLabel)
            loss.backward()
            optimizer.step()

        model.CurrentValidationLoss.clear()
        validationLoader = torch.utils.data.DataLoader(ValidationData, batch_size=250, shuffle=True, num_workers=8)
        for batch in validationLoader:
            batchLabel = batch[:,0]
            batchData = batch[:,1:].reshape(250,1,28,28).float()
            y_pred = model.forward(batchData)
            loss = loss_fn(y_pred, batchLabel)
            model.CurrentValidationLoss.append(float(loss.item()))

        model.OverallValidationLoss.append(np.mean(np.array(model.CurrentValidationLoss)))
        model.EvaluateNextEpoch()
        print(np.mean(np.array(model.CurrentValidationLoss)))

if Testing:
    resultlist = []
    testLoader = torch.utils.data.DataLoader(TestingData, batch_size=1, num_workers=8)
    for batch in testLoader:
        batchLabel = batch[:,0]
        batchData = batch[:,1:].reshape(1,1,28,28).float()
        y_pred = model.predict(batchData)
        resultlist.append(y_pred == batchLabel[0])

    resultlist = np.array(resultlist)
    total = len(resultlist)
    tp = len(resultlist[resultlist == True])
    fp = len(resultlist[resultlist == False])
    print("Total test samples: " + str(total))
    print("Correctly Predicted Samples: " + str(tp))
    print("Falsely Predicted Samples: " + str(fp))
    print("Overall Accuracy: " + str(tp/total * 100) + "%")
    print("Overall Error: " + str(fp/total * 100) + "%")
