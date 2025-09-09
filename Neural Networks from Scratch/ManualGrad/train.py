import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import nn
import optim

network = nn.NeuralNetwork()

network.add(nn.Linear(784,512))
network.add(nn.ReLU())
network.add(nn.Linear(512,256))
network.add(nn.ReLU())
network.add(nn.Linear(256,128))
network.add(nn.ReLU())
network.add(nn.Linear(128,10))
print(network)

### Prep Dataset ###
train_dataset = MNIST("../../data", train=True, download=True)
test_dataset = MNIST("../../data", train=True, download=True)

def collate_fn(batch):

    ### Prep and Scale Images ###
    images = np.concatenate([np.array(i[0]).reshape(1,784)for i in batch]) / 255

    ### Labels ###
    labels = np.array([i[1] for i in batch])

    return images, labels

trainloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
testloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=1e-3)

training_iterations = 2500

all_train_losses = []
all_train_accs = []
all_test_losses = []
all_test_accs = []

num_iters = 0
train = True
pbar = tqdm(range(training_iterations))

while train:

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for images, labels in trainloader:
        
        ### Get Outputs from Model ###
        output = network(images)
        loss = loss_func.forward(logits=output, y_true=labels)
        loss_grad = loss_func.backward()

        ### Compute Gradients in Model ###
        network.backward(loss_grad)

        ### Update Model ###
        optimizer.step()
        optimizer.zero_grad()

        ### Compute Accuracy ###
        preds = output.argmax(axis=-1)
        acc = (preds == labels).sum() / len(preds)

        ### Store Loss for Plotting ###
        train_losses.append(loss)
        train_accs.append(acc)
        
        ### Eval Loop ###
        if num_iters % 250 == 0:

            for images, labels in testloader:

                ### Get Outputs from Model ###
                output = network(images)
                loss = loss_func.forward(logits=output, y_true=labels)

                ### Compute Accuracy ###
                preds = output.argmax(axis=-1)
                acc = (preds == labels).sum() / len(preds)

                ### Store Loss for Plotting ###
                test_losses.append(loss)
                test_accs.append(acc)

            ### Average Up Performance and Store ###
            train_losses = np.mean(train_losses)
            train_accs = np.mean(train_accs)
            test_losses = np.mean(test_losses)
            test_accs = np.mean(test_accs)

            all_train_losses.append(train_losses)
            all_train_accs.append(train_accs)
            all_test_losses.append(test_losses)
            all_test_accs.append(test_accs)

            print("Training Loss:", train_losses)
            print("Training Acc:", train_accs)
            print("Testing Loss:", test_losses)
            print("Testing Acc:", test_accs)

            ### Reset Lists ###
            train_losses = []
            train_accs = []
            test_losses = []
            test_accs = []

        
        num_iters += 1
        pbar.update(1)

        if num_iters >= training_iterations:
            print("Completed Training")
            train = False
            break
