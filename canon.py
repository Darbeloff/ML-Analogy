# -*- coding: utf-8 -*-
import numpy as np
import torch, time
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
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, H = 20, 256

def f(x,y):
	out = (1-x)**2+100*(y-x**2)**2
	if (x-1)**3-y+1>0 or x+y-2>0:
		out = float('nan')
	return out
x_min, x_max, y_min, y_max = -1.5, 1.5, -0.5, 2.5
density = 50

xx,yy = np.meshgrid(np.linspace(x_min,x_max,density), np.linspace(y_min,y_max,density))
zz = np.zeros(np.shape(xx))
for xi in range(density):
	for yi in range(density):
		zz[xi,yi] = f(xx[xi,yi], yy[xi,yi])
x_lin = xx.flatten()
y_lin = yy.flatten()
z_lin = zz.flatten()
zz = np.nan_to_num(zz,nan=np.nanmax(z_lin))
z_lin = np.nan_to_num(z_lin,nan=np.nanmax(z_lin))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx,yy,zz, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(azim=0, elev=90)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
ax.set_zticks([])
# plt.show()
# exit(0)

z_lin-=np.min(z_lin)
z_lin/=np.max(z_lin)

def train_model(x,y):
	if len(np.shape(x))==1:
		x = x[:, None]
	if len(np.shape(y))==1:
		y = y[:, None]
	D_in = np.shape(x)[1]
	D_out = np.shape(y)[1]

	dataset = TensorDataset(x, y)

	N_train = int(4*len(y)/5)
	train_dataset, val_dataset = random_split(dataset, [N_train,len(y)-N_train])

	train_loader = DataLoader(dataset=train_dataset, batch_size=10)
	val_loader = DataLoader(dataset=val_dataset, batch_size=10)

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
	learning_rate = .002
	n_epochs = 300
	training_losses = []
	validation_losses = []
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	for t in range(n_epochs):
		batch_losses = []

		with torch.no_grad():
			val_losses = []
			for x_val, y_val in val_loader:
				x_val = x_val.to(device)
				y_val = y_val.to(device)
				yhat = model(x_val)
				val_loss = loss_fn(y_val, yhat).item()
				val_losses.append(val_loss)
			validation_loss = np.mean(val_losses)
			validation_losses.append(validation_loss)

		for x_batch, y_batch in train_loader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			# Forward pass: compute predicted y by passing x to the model.
			y_pred = model(x_batch)

			# Compute and print loss.
			loss = loss_fn(y_pred, y_batch)

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

		# if validation_loss<1e-2:
		# 	break
	
	# plt.figure()
	# plt.semilogy(range(len(training_losses)), training_losses, label='Training Loss')
	# plt.semilogy(range(len(training_losses)), validation_losses, label='Validation Loss')
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.legend()
	# plt.show()

	model.eval()
	return model

x = np.concatenate((np.transpose(x_lin), np.transpose(y_lin)))
y = np.transpose(z_lin)
model = train_model(x,y)
model.train()
plt.figure()
plt.plot(x,y,'k.',label='Data')

## Test model
# x_test = np.linspace(-2,2,40)
# y_test = []
# for x in x_test:
# 	y_test.append(model(torch.tensor([[x]]).float()).item())
# plt.plot(x_test, y_test,'r')
# plt.show()
# exit(0)

## Controller
n_epochs = 50
loss_fn = torch.nn.MSELoss(reduction='sum')
x0 = -1.8
inp = Variable(torch.tensor([[x0+0.0]]).type(dtype), requires_grad=True)
# gradient_mask = torch.zeros(1,1)
# gradient_mask[0,0] = 1.0
# inp.register_hook(lambda grad: grad.mul_(gradient_mask))
optimizer = torch.optim.SGD([inp], lr=.2)
opt_steps_x = []
opt_steps_y = []
for t in range(n_epochs):
	y0 = model(inp)
	loss = loss_fn(y0, torch.tensor([[y0.item()-0.1]]))
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	opt_steps_x.append(inp.data[0,0]+0.0)
	opt_steps_y.append(f(inp.data[0,0]+0.0))

plt.plot(opt_steps_x, opt_steps_y, 'r+', label='Steps')
plt.plot(opt_steps_x[np.argmin(opt_steps_y)], min(opt_steps_y), 'g*', label='Final')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(loc='upper center')
plt.tight_layout()
plt.show()