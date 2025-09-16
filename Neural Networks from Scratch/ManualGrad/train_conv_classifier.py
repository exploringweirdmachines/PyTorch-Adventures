import cupy as np ### USING NP NOTATION BUT CUPY ALMOST IDENTICAL TO NUMPY
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import nn
import optim
from torchvision import transforms

network = nn.NeuralNetwork()

# First block
network.add(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False))
network.add(nn.BatchNorm2d(num_features=32))
network.add(nn.ReLU())

network.add(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False))
network.add(nn.BatchNorm2d(num_features=64))
network.add(nn.ReLU())

# Second block
network.add(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
network.add(nn.BatchNorm2d(num_features=128))
network.add(nn.ReLU())

network.add(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False))
network.add(nn.BatchNorm2d(num_features=128))
network.add(nn.ReLU())

# Third block
network.add(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False))
network.add(nn.BatchNorm2d(num_features=256))
network.add(nn.ReLU())

network.add(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False))
network.add(nn.BatchNorm2d(num_features=256))
network.add(nn.ReLU())

# Classifier
network.add(nn.Flatten())
network.add(nn.Dropout(p=0.2))
network.add(nn.Linear(256*4*4, 10))

print(network)

### Prep Dataset ###
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], 
        std=[0.2470, 0.2435, 0.2616]
    )
])

train_dataset = CIFAR10("../../data", train=True, download=True, transform=transform)
test_dataset  = CIFAR10("../../data", train=False, download=True, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2470, 0.2435, 0.2616]
                            )
                        ]))

def collate_fn(batch):
    images = np.stack([np.array(i[0], dtype=np.float32) for i in batch]) / 255.0
    labels = np.array([i[1] for i in batch])
    return images, labels

trainloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
testloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=5e-4)

training_iterations = 10000

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
        train_losses.append(loss.get())
        train_accs.append(acc.get())
        
        ### Eval Loop ###
        if num_iters % 250 == 0:

            for images, labels in tqdm(testloader):

                ### Get Outputs from Model ###
                output = network(images)
                loss = loss_func.forward(logits=output, y_true=labels)
                
                ### Compute Accuracy ###
                preds = output.argmax(axis=-1)
                acc = (preds == labels).sum() / len(preds)

                ### Store Loss for Plotting ###
                test_losses.append(loss.get())
                test_accs.append(acc.get())

            ### Average Up Performance and Store ###
            train_losses = np.mean(np.array(train_losses)).get()
            train_accs = np.mean(np.array(train_accs)).get()
            test_losses = np.mean(np.array(test_losses)).get()
            test_accs = np.mean(np.array(test_accs)).get()

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
