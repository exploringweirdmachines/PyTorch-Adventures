import os
import cupy as cp
import numpy as np
from tqdm import tqdm
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from myaccelerator.accelerator import Accelerator

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()

########################
### LOAD ACCELERATOR ###
########################
accelerator = Accelerator()

####################
### DEFINE MODEL ###
####################

class MyTorchMNIST(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(784, 512)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.drop3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.drop1(self.activation(self.fc1(x)))
        x = self.drop2(self.activation(self.fc2(x)))
        x = self.drop3(self.activation(self.fc3(x)))
        x = self.fc4(x)
        return x

model = MyTorchMNIST()
model = accelerator.prepare_model(model)

####################
### LOAD DATASET ###
####################
train = MNIST("../../data", train=True, download=False)
test = MNIST("../../data", train=False, download=False)

def collate_fn(batch):
    images = cp.concatenate([cp.array(i[0]).astype(cp.float32).reshape(1,784) for i in batch]) / 255
    labels = [i[1] for i in batch]
    images = mytorch.Tensor(images)
    labels = mytorch.Tensor(labels)
    return images, labels

trainloader = DataLoader(train, batch_size=16, collate_fn=collate_fn)
testloader = DataLoader(test, batch_size=16, collate_fn=collate_fn)

###############################
### LOAD OPTIMIZER AND LOSS ###
###############################
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

#############
### TRAIN ###
#############

NUM_EPOCHS = 5

for epoch in range(NUM_EPOCHS):
    
    accelerator.print(f"Epoch {epoch}")

    train_loss, train_acc = [], []

    model.train()
    for images, labels in tqdm(trainloader, disable=not accelerator.is_main_process()):

        # Forward
        pred = model(images)
        loss = loss_fn(pred, labels)

        # Backward
        accelerator.backward(loss)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Compute accuracy
        predicted = pred.argmax(dim=-1)
        accuracy = (predicted == labels).sum() / len(predicted)
        train_loss.append(loss.item())
        train_acc.append(accuracy.item())

    accelerator.print(f"Train Loss: {np.mean(train_loss):.4f}, Train Acc: {np.mean(train_acc):.4f}")

accelerator.end_training()