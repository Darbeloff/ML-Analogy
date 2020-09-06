# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dtype = torch.FloatTensor
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, H, nc = 14, 256, 2

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

	train_dataset, val_dataset = random_split(dataset, [int(N/2),int(N/2)])

	train_loader = DataLoader(dataset=train_dataset, batch_size=D_in)
	val_loader = DataLoader(dataset=val_dataset, batch_size=D_in)

	# Use the nn package to define our model and loss function.
	# model = torch.nn.Sequential(
	# 	torch.nn.Linear(D_in, H),
	# 	torch.nn.ReLU(),
	# 	torch.nn.ReLU(),
	# 	torch.nn.ReLU(),
	# 	torch.nn.Linear(H, D_out),
	# )
	# model = torch.nn.Linear(D_in, D_out, bias=False)
	model_w = Variable(torch.randn(D_in, D_out).type(dtype), requires_grad=True)
	loss_fn = torch.nn.MSELoss(reduction='sum')

	# Use the optim package to define an Optimizer that will update the weights of
	# the model for us. Here we will use Adam; the optim package contains many other
	# optimization algorithms. The first argument to the Adam constructor tells the
	# optimizer which Tensors it should update.
	learning_rate = .02
	n_epochs = 10000
	training_losses = []
	validation_losses = []
	for t in range(n_epochs):
		batch_losses = []
		for x_batch, y_batch in train_loader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			# Forward pass: compute predicted y by passing x to the model.
			y_pred = x_batch.mm(model_w)

			# Compute and print loss.
			loss = loss_fn(y_pred, y_batch)

			# Backward pass: compute gradient of the loss with respect to model
			# parameters
			loss.backward()

			# Calling the step function on an Optimizer makes an update to its
			# parameters
			model_w.data -= learning_rate*model_w.grad.data

			model_w.grad.data.zero_()

			batch_losses.append(loss.item())
		training_loss = np.mean(batch_losses)
		training_losses.append(training_loss)

		with torch.no_grad():
			val_losses = []
			for x_val, y_val in val_loader:
				x_val = x_val.to(device)
				y_val = y_val.to(device)
				yhat = x_val.mm(model_w)
				val_loss = loss_fn(y_val, yhat).item()
				val_losses.append(val_loss)
			validation_loss = np.mean(val_losses)
			validation_losses.append(validation_loss)

		print(f"[{t+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

		if validation_loss<1e-5:
			break
	
	# plt.figure()
	# plt.semilogy(range(len(training_losses)), training_losses, label='Training Loss')
	# plt.semilogy(range(len(training_losses)), validation_losses, label='Validation Loss')
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.legend()
	# plt.show()
	
	return model_w

def u2th(u_t, u_tm1, u_tm2, th_tm1, th_tm2):
	return (l*T*T*(u_t+2*u_tm1+u_tm2) -(-8*I+2*L*L*k*T*T)*th_tm1 -(4*I-2*L*L*c*T+L*L*k*T*T)*th_tm2) / (4*I+2*L*L*c*T+L*L*k*T*T)

def controller(model, u_tm1, u_tm2, th_tm1, th_tm2, ref):
	loss_fn = torch.nn.MSELoss(reduction='sum')
	n_epochs = 100
	learning_rate = 0.5
	u_t = u_tm1
	for t in range(n_epochs):
		inp = Variable(torch.tensor([[u_t, u_tm1, u_tm2, th_tm1, th_tm2]]).type(dtype), requires_grad=True)
		x_t = inp.mm(model)
		loss = loss_fn(x_t, torch.tensor([[ref]]).float())
		loss.backward()
		u_t -= learning_rate*inp.grad.data[0][0]
		inp.grad.data.zero_()

	return u_t

# Create random Tensors to hold inputs and outputs
x = 2*np.random.rand(N, 5) # [u[t], u[t-1], u[t-2], f[t-1], f[t-2]]
model = train_model(x, u2th(x[:,0], x[:,1], x[:,2], x[:,3], x[:,4]))

def reference(t):
	return 1 # Unit Step
	#return np.sin(t)

t_vec = np.arange(0, 10, T)
# u_vec = [0,0]
# x_vec = [0,0]
# x_vec_sim = []
# for t in t_vec:
# 	u_vec.append(reference(t))
# 	x_vec_sim.append( torch.tensor([[u_vec[-1], u_vec[-2], u_vec[-3], x_vec[-1], x_vec[-2]]]).float().mm(model) )
# 	x_vec.append(u2th(u_vec[-1], u_vec[-2], u_vec[-3], x_vec[-1], x_vec[-2]))
# plt.figure()
# plt.plot(t_vec, u_vec[2:], label='u')
# plt.plot(t_vec, x_vec[2:], label='x')
# plt.plot(t_vec, x_vec_sim, label='x_sim')
# plt.xlabel('Time')
# plt.title('Response to Unit Step Input')
# plt.legend()
# plt.show()

u_vec = [0,0]
x_vec = [0,0]
for t in t_vec:
	u_vec.append(controller(model, u_vec[-1], u_vec[-2], x_vec[-1], x_vec[-2], 1))
	x_vec.append(u2th(u_vec[-1], u_vec[-2], u_vec[-3], x_vec[-1], x_vec[-2]))
plt.figure()
plt.plot(t_vec, u_vec[2:], label='u')
plt.plot(t_vec, x_vec[2:], label='x')
plt.xlabel('Time')
plt.title('Response to Unit Step Input')
plt.legend()
plt.show()