import os
import cupy as cp
import numpy as np
from tqdm import tqdm
import mytorch
import mytorch.nn as nn
import mytorch.optim as optim
from torchvision.datasets import MNIST
from mytorch.data import DataLoader
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


####################
### LOAD DATASET ###
####################
train = MNIST("../../data", train=True, download=False)
test = MNIST("../../data", train=False, download=False)

def collate_fn(batch):
    images = np.concatenate([np.array(i[0]).astype(np.float32).reshape(1,784) for i in batch]) / 255
    labels = np.array([i[1] for i in batch])
    return images, labels

trainloader = DataLoader(train, batch_size=16, collate_fn=collate_fn, num_workers=2)
testloader = DataLoader(test, batch_size=16, collate_fn=collate_fn, num_workers=2)

###############################
### LOAD OPTIMIZER AND LOSS ###
###############################
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

##########################
### PREPARE EVERYTHING ###
##########################
model, optimizer, trainloader, testloader = accelerator.prepare(
    model, optimizer, trainloader, testloader
) 
# trainloader = accelerator.prepare_dataloaders(trainloader)
# testloader = accelerator.prepare_dataloaders(testloader)
# optimizer = accelerator.prepare_optimizer(optimizer)
# model = accelerator.prepare_model(model)
#############
### TRAIN ###
#############

NUM_EPOCHS = 5

for epoch in range(NUM_EPOCHS):
    
    accelerator.print(f"Epoch {epoch}")

    train_loss, train_acc = [], []
    eval_loss, eval_acc = [], []

    model.train()
    for images, labels in tqdm(trainloader, disable=not accelerator.is_main_process()):
        
        ### Convert Numpy Arrays to CuPY Arrays on GPU ###
        images = mytorch.Tensor(images)
        labels = mytorch.Tensor(labels)

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

        train_loss.append(accelerator.gather_for_metrics(loss))
        train_acc.append(accelerator.gather_for_metrics(accuracy))

    model.eval()
    for images, labels in tqdm(testloader):
        
        ### Convert Numpy Arrays to CuPY Arrays on GPU ###
        images = mytorch.Tensor(images)
        labels = mytorch.Tensor(labels)

        with mytorch.no_grad():
            ### Pass Through Model ###
            pred = model(images)
            
            ### Compute Loss ###
            loss = loss_fn(pred, labels)

        ### Compute Accuracy ###
        predicted = pred.argmax(dim=-1)
        accuracy = (predicted == labels).sum() / len(predicted)

        eval_loss.append(accelerator.gather_for_metrics(loss))
        eval_acc.append(accelerator.gather_for_metrics(accuracy))

    accelerator.print(f"Train Loss: {np.mean(train_loss):.4f}, Train Acc: {np.mean(train_acc):.4f}")
    accelerator.print(f"Eval Loss: {np.mean(eval_loss):.4f}, Eval Acc: {np.mean(eval_acc):.4f}")

accelerator.end_training()