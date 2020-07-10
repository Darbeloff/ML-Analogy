# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1000, 3, 256, 1

# Create random Tensors to hold inputs and outputs
k = 1
x = np.random.rand(N, D_in) # [F, L, theta]
y = 10*x[:,0]*x[:,1]-k*x[:,2]
#x = np.delete(x,0,1); D_in = 2;
y = y[:, None]

x_train_tensor = torch.from_numpy(x).float()
y_train_tensor = torch.from_numpy(y).float()

dataset = TensorDataset(x_train_tensor, y_train_tensor)

train_dataset, val_dataset = random_split(dataset, [int(0.8*N), int(0.2*N)])

train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
n_epochs = 100
training_losses = []
validation_losses = []
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(n_epochs):
    batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        model.train()
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x_batch)

        # Compute and print loss.
        loss = loss_fn(y_pred, y_batch)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        batch_losses.append(loss.item())
    training_loss = np.mean(batch_losses)
    training_losses.append(training_losses)

    with torch.no_grad():
        val_losses = []
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            model.eval()
            yhat = model(x_val)
            val_loss = loss_fn(y_val, yhat).item()
            val_losses.append(val_loss)
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    print(f"{t+1}\tTraining loss\t{training_loss:.3f}\tValidation loss\t{validation_loss:.3f}")