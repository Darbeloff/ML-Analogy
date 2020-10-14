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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.FloatTensor
torch.manual_seed(6)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, H = 20, 256

p_in_data = np.array([])
spd_data = np.array([])
eff_data = np.array([])
with open('esc.csv') as file:
	reader = csv.DictReader(file)
	for row in reader:
		p_in_data = np.append(p_in_data, float(row['Input_Power']))
		spd_data = np.append(spd_data, float(row['Speed']))
		eff_data = np.append(eff_data, float(row['Efficiency']))

p_in_data-=np.min(p_in_data)
spd_data-=np.min(spd_data)
eff_data-=np.min(eff_data)

p_in_data/=np.max(p_in_data)
spd_data/=np.max(spd_data)
eff_data/=np.max(eff_data)

x = np.concatenate((np.transpose(p_in_data[0:-1][None]), np.transpose(spd_data[0:-1][None])),1)
y = np.concatenate((np.transpose(spd_data[1:][None]), np.transpose(eff_data[1:][None])),1)
# fig = plt.figure()
# plt.plot(x[:,0], x[:,1],'r.')
# plt.show()
# exit()

triObj = Triangulation(x[:,0], x[:,1])
pre_plant_s = LinearTriInterpolator(triObj,y[:,0])
pre_plant_p = LinearTriInterpolator(triObj,y[:,1])
def plant_s(p_in, spd):
	out = pre_plant_s(p_in, spd)
	# if np.isnan(out):
	# 	breakpoint()
	# 	out = 1
	return out
def plant_p(p_in, spd):
	out = pre_plant_p(p_in, spd)
	# if np.isnan(out):
	# 	out = 1
	return out

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot(x[:,0], x[:,1], y[:,1],'r.')
min_i = np.argmax(y[:,1])
ax1.plot([x[min_i,0]],[x[min_i,1]],[y[min_i,1]],'g*',label='True Extremum',zorder=1e3)
plt.xlabel('Input Power')
plt.ylabel('Speed')
# plt.title('Efficiency')
ax1.view_init(azim=0, elev=90)

ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.plot(x[:,0], x[:,1], y[:,0],'r.')
min_i = np.argmax(y[:,1])
ax2.plot([x[min_i,0]],[x[min_i,1]],[y[min_i,1]],'g*',label='True Extremum',zorder=1e3)
plt.xlabel('Input Power')
plt.ylabel('Speed')
# plt.title('Next Speed')
ax2.view_init(azim=0, elev=90)

# xs = np.linspace(0,1,200)
# ys = np.linspace(0,1,200)
# xs,ys = np.meshgrid(xs,ys)
# zs = np.zeros(np.shape(xs))
# zp = np.zeros(np.shape(xs))
# for xi in range(np.shape(xs)[0]):
# 	for yi in range(np.shape(xs)[1]):
# 		zs[xi,yi] = plant_s(xs[xi,yi],ys[xi,yi])
# 		zp[xi,yi] = plant_p(xs[xi,yi],ys[xi,yi])
# zs = np.nan_to_num(zs)
# zp = np.nan_to_num(zp)
# ax1.plot_surface(xs,ys,zp, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# ax2.plot_surface(xs,ys,zs, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
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
	n_epochs = 5000
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
		
		if t>1000 and validation_losses[-1]<0.05 and np.mean(validation_losses[-20:-10])<np.mean(validation_losses[-9:-1]):
			break

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

xs = np.linspace(0,1,200)
ys = np.linspace(0,1,200)
xs,ys = np.meshgrid(xs,ys)
z_p = np.zeros(np.shape(xs))
z_s = np.zeros(np.shape(xs))
for xi in range(np.shape(xs)[0]):
	for yi in range(np.shape(xs)[1]):
		modelOut = model(torch.tensor([[xs[xi,yi],ys[xi,yi]]]).type(dtype))
		z_p[xi,yi] = modelOut[0,1].item()
		z_s[xi,yi] = modelOut[0,0].item()
ax1.plot_surface(xs,ys,z_p, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax2.plot_surface(xs,ys,z_s, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# plt.show()
# exit()

DEBUG = False
# DEBUG = True

t_vec = range(100)
p_in_vec = [0.8]
spd_vec = [0.6]
p_out_vec = []
p_out_sim = []
spd_sim = []
if DEBUG:
	ax1.plot([p_in_vec[-1]], [spd_vec[-1]], [1], 'k+', zorder=1e3)
	ax2.plot([p_in_vec[-1]], [spd_vec[-1]], [1], 'k+', zorder=1e3)

## Controller
n_epochs = 70
loss_fn = torch.nn.MSELoss(reduction='sum')
inp = Variable(torch.tensor([[p_in_vec[-1]+0.0, spd_vec[-1]+0.0]]).type(dtype), requires_grad=True)
gradient_mask = torch.zeros(1,2)
gradient_mask[0,0] = 1.0
inp.register_hook(lambda grad: grad.mul_(gradient_mask))
optimizer = torch.optim.Adam([inp], lr=.002)
for t in range(n_epochs):
	y0 = model(inp)
	loss = loss_fn(y0[0,1], torch.tensor(y0[0,1].item()+0.1))
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if DEBUG:
		ax1.plot([p_in_vec[-1]], [spd_vec[-1]], [1], 'g+', zorder=1e3)
		ax2.plot([p_in_vec[-1]], [spd_vec[-1]], [1], 'g+', zorder=1e3)

	p_out_sim.append(y0[0,1].item())
	spd_sim.append(y0[0,0].item())
	p_in_vec.append(inp[0,0].item())
	spd_vec.append(plant_s(p_in_vec[-1], spd_vec[-1]).data.item())
	inp[0,1].data.copy_(torch.tensor(spd_vec[-1]))
	p_out_vec.append(plant_p(p_in_vec[-1], spd_vec[-1]).data.item())

	# ax.plot([p_in_vec[-1]], [spd_vec[-1]], [p_out_vec[-1]], 'g+')
	# ax.plot([p_in_vec[-1]], [spd_vec[-1]], [p_out_sim[-1]], 'b+')

	if DEBUG:
		ax1.plot([p_in_vec[-1]], [spd_vec[-1]], [1], 'm+', zorder=1e3, label='Steps')
		ax2.plot([p_in_vec[-1]], [spd_vec[-1]], [1], 'm+', zorder=1e3, label='Steps')
		plt.show(block=False)
		breakpoint()

ax1.plot(p_in_vec, spd_vec, np.ones(np.size(p_in_vec)), 'k+', zorder=1e3, label='Steps')
ax2.plot(p_in_vec, spd_vec, np.ones(np.size(p_in_vec)), 'k+', zorder=1e3, label='Steps')

ax1.set_xlim([0.6,1])
ax1.set_ylim([0.5,1])
ax2.set_xlim([0.6,1])
ax2.set_ylim([0.5,1])

ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
plt.tight_layout()
plt.legend(loc='lower center', ncol=2)
plt.show()