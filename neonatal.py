# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1000, 5, 256, 1

k = 1          # spring constant
T = 1e-3       #[s] sampling period
m = 1          #[kg] mass
c = 1          # damper
l = 0.5        # moment arm of muscle
L = 1          # moment arm of spring

def haptic_feedback(f_tm1, f_tm2, u_t, u_tm1, u_tm2):
	return (3*k*T*T*u_t + 6*k*T*T*u_tm1 + 3*k*T*T*u_tm2 - 2*l*(k*T*T-4*m*L)*f_tm1 - l*(4*m*L+k*T*T)*f_tm2) / (l*(4*m*L+k*T*T))

def controller(e):
	return k*e

def reference(t):
	return 1 # Unit Step
	#return np.sin(t)
'''
t_vec = np.arange(0,10,T)
f_vec = [0,0]
u_vec = [0,0]
for t in t_vec:
	u_vec.append(1)
	f_vec.append(haptic_feedback(f_vec[-1], f_vec[-2], u_vec[-1], u_vec[-2], u_vec[-3]))
f_vec.pop(0)
f_vec.pop(0)
u_vec.pop(0)
u_vec.pop(0)

plt.figure()
plt.plot(t_vec,f_vec)
plt.show()
exit(0)
'''
# Create random Tensors to hold inputs and outputs
x = 2*(2*np.random.rand(N, D_in)-1)
y = haptic_feedback(x[:,0], x[:,1], x[:,2], x[:,3], x[:,4])
y = y[:, None]

x_train_tensor = torch.from_numpy(x).float()
y_train_tensor = torch.from_numpy(y).float()

dataset = TensorDataset(x_train_tensor, y_train_tensor)

train_dataset, val_dataset = random_split(dataset, [int(0.9*N), N-int(0.9*N)])

train_loader = DataLoader(dataset=train_dataset, batch_size=16)
val_loader = DataLoader(dataset=val_dataset, batch_size=20)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
	#torch.nn.Linear(D_in, H, bias=False),
	torch.nn.LSTM(input_size=D_in,hidden_size=3),
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
	training_losses.append(training_loss)

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
		if t>2 and validation_losses[-1] >= validation_losses[-2]:
			n_epochs = t+1
			break

	print(f"[{t+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

model.eval()
'''
plt.figure()
plt.semilogy(range(n_epochs), training_losses, label='Training Loss')
plt.semilogy(range(n_epochs), validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
for x_batch, y_batch in train_loader:
	in_tensor = x_batch[0]
	break
def nn_plant(f_tm1, f_tm2, u_t, u_tm1, u_tm2):
	in_tensor[0] = f_tm1
	in_tensor[1] = f_tm2
	in_tensor[2] = u_t
	in_tensor[3] = u_tm1
	in_tensor[3] = u_tm2
	out = model(in_tensor)[0].item()
	return out

t_vec = np.arange(0, 10, T)
u_vec = [0,0]
f_vec = [0,0,0]
f_sim_vec = [0]
for t in t_vec:
	u_vec.append(1)

	f_sim_vec.append(nn_plant(f_vec[-1], f_vec[-2], u_vec[-1], u_vec[-2], u_vec[-3]))

	f_vec.append(haptic_feedback(f_vec[-1], f_vec[-2], u_vec[-1], u_vec[-2], u_vec[-3]))

u_vec.pop(0)
u_vec.pop(0)
f_vec.pop(0)
f_vec.pop(0)

plt.figure()
plt.plot(t_vec, u_vec, label='u')
plt.plot(t_vec,f_vec[:-1], label='f')
plt.plot(t_vec,f_sim_vec[:-1], label='f_sim')
plt.xlabel('Time')
plt.title('Response to Unit Step Input')
plt.legend()
plt.show()