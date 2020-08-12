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
N, H, D_out = 1000, 256, 1

k = 1          # stiffness
T = 1e-2       #[s] sampling period
m = 1          #[kg] mass
c = 1          # damper
I = 1          # rotational inertia
l = 0.5        # distance to muscle attachement
L = 1          # length of arm

def train_model(x,y):
	if len(np.shape(y))==1:
		y = y[:, None]
	D_in = np.shape(x)[1]
	D_out = np.shape(y)[1]

	x_train_tensor = torch.from_numpy(x).float()
	y_train_tensor = torch.from_numpy(y).float()

	dataset = TensorDataset(x_train_tensor, y_train_tensor)

	train_dataset, val_dataset = random_split(dataset, [int(0.8*N), N-int(0.8*N)])

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
	n_epochs = 10000
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

		print(f"[{t+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
		if t>2 and validation_losses[-1] >= validation_losses[-2]:
			break

	model.eval()

	plt.figure()
	plt.semilogy(range(len(training_losses)), training_losses, label='Training Loss')
	plt.semilogy(range(len(training_losses)), validation_losses, label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

	return model

def u2F(u_t, u_tm1, u_tm2, f_tm1, f_tm2):
	return ((2*c*L*l*T+k*l*L*T*T)*u_t + 2*k*l*L*T*T*u_tm1 + (-2*c*L*l*T+k*l*L*T*T)*u_tm2 - (-8*I+2*k*L*T*T)*f_tm1 - (4*I-2*c*L*T+k*L*T*T)*f_tm2) / (4*I+2*c*L*T+k*L*T*T)

def uF2th(u_t, u_tm1, u_tm2, f_t, f_tm1, f_tm2, th_tm1, th_tm2):
	return (l*u_t +2*l*u_tm1 +l*u_tm2 -f_t -2*f_tm1 -f_tm2 +8*I/T/T*th_tm1 -4*I/T/T*th_tm2)*T*T/4/I

def end2end(u_t, u_tm1, u_tm2, f_tm1, f_tm2, th_tm1, th_tm2):
	f_t = u2F(u_t, u_tm1, u_tm2, f_tm1, f_tm2)
	th_t = uF2th(u_t, u_tm1, u_tm2, f_t, f_tm1, f_tm2, th_tm1, th_tm2)
	return np.concatenate((f_t[:,None],th_t[:,None]), 1)

# Create random Tensors to hold inputs and outputs
x = 2*np.random.rand(N, 5) # [u[t], u[t-1], u[t-2], f[t-1], f[t-2]]
stage_1 = train_model(x, u2F(x[:,0], x[:,1], x[:,2], x[:,3], x[:,4]))

x = 2*np.random.rand(N, 8) # [u[t], u[t-1], u[t-2], f[t], f[t-1], f[t-2], theta[t-1], theta[t-2]]
stage_2 = train_model(x, uF2th(x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6], x[:,7]))

x = 2*np.random.rand(N, 7) # [u[t], u[t-1], u[t-2], f[t-1], f[t-2], theta[t-1], theta[t-2]]
e2e = train_model(x, end2end(x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5], x[:,6]))

'''
for x_batch, y_batch in train_loader:
	in_tensor = x_batch[0]
	break
def nn_plant(u_t, u_tm1, u_tm2, f_tm1, f_tm2):
	in_tensor[0] = u_t
	in_tensor[1] = u_tm1
	in_tensor[2] = u_tm2
	in_tensor[3] = f_tm1
	in_tensor[3] = f_tm2
	return model(in_tensor)[0].item()

def reference(t):
	return 1 # Unit Step
	#return np.sin(t)

t_vec = np.arange(0, 10, T)
u_vec = [0,0]
F_vec = [0,0]
F_vec_sim = []
for t in t_vec:
	u_vec.append(reference(t))
	F_vec_sim.append(nn_plant(u_vec[-1], u_vec[-2], u_vec[-3], F_vec[-1], F_vec[-2]))
	F_vec.append(u2F(u_vec[-1], u_vec[-2], u_vec[-3], F_vec[-1], F_vec[-2]))

plt.figure()
plt.plot(t_vec, u_vec[2:], label='u')
plt.plot(t_vec, F_vec[2:], label='F')
plt.plot(t_vec, F_vec_sim, label='F_sim')
plt.xlabel('Time')
plt.title('Response to Unit Step Input')
plt.legend()
plt.show()
'''