# -*- coding: utf-8 -*-
import csv
import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator

import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.FloatTensor

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, H = 20, 256

x = []
y = []
with open('esc.csv') as file:
	reader = csv.DictReader(file)
	first = True
	for row in reader:
		if first:
			first = False
		else:
			x.append([float(row['Input_Power']), float(lastSpeed)])
			y.append([float(row['Speed']), float(row['Output_Power'])])
		lastSpeed = row['Speed']

x = np.array(x)
y = np.array(y)

x_shift = x.min(axis=0)
y_shift = y.min(axis=0)
for i in range(len(x)):
	x[i]-=x_shift
	y[i]-=y_shift
x_transform = x.max(axis=0)
y_transform = y.max(axis=0)
for i in range(len(x)):
	x[i]/=x_transform
	y[i]/=y_transform

triObj_p = Triangulation(x[:,0], x[:,1]) 
plant_p = LinearTriInterpolator(triObj_p,y[:,0])
triObj_s = Triangulation(x[:,0], x[:,1]) 
plant_s = LinearTriInterpolator(triObj_s,y[:,1])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x[:,0], x[:,1], y[:,0],'r.')
# # plt.show()
# xs = np.linspace(0,1,200)
# ys = np.linspace(0,1,200)
# xs,ys = np.meshgrid(xs,ys)
# zs = np.zeros(np.shape(xs))
# for xi in range(np.shape(xs)[0]):
# 	for yi in range(np.shape(xs)[1]):
# 		zs[xi,yi] = plant_p(xs[xi,yi],ys[xi,yi])
# surf = ax.plot_surface(xs,ys,zs)
# plt.show()
# exit(0)

def train_model(x,y):
	if len(np.shape(y))==1:
		y = y[:, None]
	D_in = np.shape(x)[1]
	D_out = np.shape(y)[1]

	x_train_tensor = torch.from_numpy(x).float()
	y_train_tensor = torch.from_numpy(y).float()

	dataset = TensorDataset(x_train_tensor, y_train_tensor)

	N_train = int(4*len(y)/5)
	train_dataset, val_dataset = random_split(dataset, [N_train,len(y)-N_train])

	train_loader = DataLoader(dataset=train_dataset, batch_size=N)
	val_loader = DataLoader(dataset=val_dataset, batch_size=N)

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
	learning_rate = .0001
	n_epochs = 500
	training_losses = []
	validation_losses = []
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	for t in range(n_epochs):

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

		print(f"[{t+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
		
		# if t>10 and validation_losses[-1]<0.09 and validation_losses[-2]<validation_losses[-1]:
		# 	break

	model.eval()
	
	# plt.figure()
	# plt.semilogy(range(len(training_losses)), training_losses, label='Training Loss')
	# plt.semilogy(range(len(training_losses)), validation_losses, label='Validation Loss')
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.legend()
	# plt.show()
	
	return model

# Create random Tensors to hold inputs and outputs
model = train_model(x, y)

t_vec = range(100)
p_in_vec = [0.2]
spd_vec = [0.9]
p_out_vec = []
p_out_sim = []
spd_sim = []

## Controller
n_epochs = 100
loss_fn = torch.nn.MSELoss(reduction='sum')
inp = Variable(torch.tensor([[p_in_vec[-1]+0.0, spd_vec[-1]+0.0]]).type(dtype), requires_grad=True)
gradient_mask = torch.zeros(1,2)
gradient_mask[0,0] = 1.0
inp.register_hook(lambda grad: grad.mul_(gradient_mask))
optimizer = torch.optim.Adam([inp], lr=.02)
for t in range(n_epochs):
	y0 = model(inp)
	loss = loss_fn(y0[0,0], torch.tensor([[y0.data[0,0]+0.1]])[0,0])
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	p_out_sim.append(y0[0,0].item())
	spd_sim.append(y0[0,1].item())
	p_in_vec.append(inp[0,0].item())
	# breakpoint()
	spd_vec.append(plant_s(p_in_vec[-1], spd_vec[-1]).data.item())
	inp[0,1] = spd_vec[-1]
	p_out_vec.append(plant_p(p_in_vec[-1], spd_vec[-1]).data.item())
	# breakpoint()
	ax.plot([p_in_vec[-1]], [spd_vec[-1]], [p_out_vec[-1]], 'r+')
	ax.plot([p_in_vec[-1]], [spd_vec[-1]], [p_out_sim[-1]], 'b+')

plt.show()