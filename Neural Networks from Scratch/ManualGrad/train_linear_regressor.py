import cupy as np ### Changed to numpy for simplicity
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import nn
import optim

network = nn.NeuralNetwork()

network.add(nn.Linear(1,256))
network.add(nn.ReLU())
network.add(nn.Linear(256,512))
network.add(nn.ReLU())
network.add(nn.Linear(512,256))
network.add(nn.ReLU())
network.add(nn.Linear(256,1))

### Synthetic Dataset for Regression with single input ###
def func(x):
    return x*np.exp(np.sin(2 * np.pi * x))

class SyntheticRegressionDataset(Dataset):
    def __init__(self, size, input_dim=1):
        self.x = np.random.uniform(-1, 1, (size, input_dim))
        self.y = func(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

### Prep Dataset ###
train_dataset = SyntheticRegressionDataset(60000)
test_dataset = SyntheticRegressionDataset(10000)

def collate_fn(batch):
    ### Stack Inputs and Targets ###
    x = np.stack([i[0] for i in batch])
    y = np.stack([i[1] for i in batch])
    return x, y

trainloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
testloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
loss_func = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=1e-3)

training_iterations = 2500

all_train_losses = []
all_train_maes = []
all_test_losses = []
all_test_maes = []

num_iters = 0
train = True
pbar = tqdm(range(training_iterations))

while train:

    train_losses = []
    train_maes = []
    test_losses = []
    test_maes = []

    for x, y in trainloader:

        ### Get Outputs from Model ###
        output = network(x)
        loss = loss_func.forward(output, y)
        loss_grad = loss_func.backward()

        ### Compute Gradients in Model ###
        network.backward(loss_grad)

        ### Update Model ###
        optimizer.step()
        optimizer.zero_grad()

        ### Store Loss for Plotting ###
        train_losses.append(loss.get())  
        
        ### Eval Loop ###
        if num_iters % 250 == 0:

            for images, labels in testloader:

                ### Get Outputs from Model ###
                output = network(images)
                loss = loss_func.forward(output, labels)

                ### Store Loss for Plotting ###
                test_losses.append(loss.get())

            ### Average Up Performance and Store ###
            avg_train_loss = np.mean(np.array(train_losses))
            avg_test_loss = np.mean(np.array(test_losses))

            all_train_losses.append(avg_train_loss)
            all_test_losses.append(avg_test_loss)

            print("Training Loss:", avg_train_loss)
            print("Testing Loss:", avg_test_loss)

            ### Reset Lists ###
            train_losses = []
            test_losses = []

        num_iters += 1
        pbar.update(1)

        if num_iters >= training_iterations:
            print("Completed Training")
            train = False
            break

### Plot Results ##
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
y_true = func(x_test)
y_pred = network(x_test)

plt.plot(x_test.get(), y_true.get(), label='True', linewidth=3,)
plt.plot(x_test.get(), y_pred.get(), label='Predicted', linewidth=3, linestyle='--')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('True vs Predicted')
plt.show()