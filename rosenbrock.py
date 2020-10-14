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
torch.manual_seed(7) # 5
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, H = 20, 256
DEBUG = False
# DEBUG = True

def f(x,y):
	out = (1-x)**2+100*(y-x**2)**2
	if x**2+y**2>2:
		out = float('nan')
	return out
x_min, x_max, y_min, y_max = -1.5, 1.5, -1.5, 1.5
density = 50
# x0, y0 = -0.5,-1.0
x0, y0 = -0.4,-1.2

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
if DEBUG:
	ax1 = fig.add_subplot(121, projection='3d')
else:
	ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(xx,yy,zz, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.view_init(azim=0, elev=90)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
ax1.set_zticks([])
# plt.show()
# exit(0)

z_lin-=np.min(z_lin)
z_lin/=np.max(z_lin)
min_i = np.argmin(z_lin)
xm = x_lin[min_i]
ym = y_lin[min_i]

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

	train_loader = DataLoader(dataset=train_dataset, batch_size=30)
	val_loader = DataLoader(dataset=val_dataset, batch_size=30)

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
	learning_rate = .001
	n_epochs = 1000
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

		if np.mean(validation_losses[-20:-10])<np.mean(validation_losses[-9:-1]):
			break
	
	# plt.figure()
	# plt.semilogy(range(len(training_losses)), training_losses, label='Training Loss')
	# plt.semilogy(range(len(training_losses)), validation_losses, label='Validation Loss')
	# plt.xlabel('Epoch')
	# plt.ylabel('Loss')
	# plt.legend()
	# plt.show()

	model.eval()
	return model

x = torch.from_numpy(np.concatenate((x_lin[:,None], y_lin[:,None]), 1)).float()
y = torch.from_numpy(z_lin[:,None]).float()

# Save/Load
# model = train_model(x,y)
# torch.save(model.state_dict(), 'rosenbrock.pt')

model = torch.nn.Sequential(
		torch.nn.Linear(2, H),
		torch.nn.ReLU(),
		torch.nn.ReLU(),
		torch.nn.ReLU(),
		torch.nn.Linear(H, 1),
	)
model.load_state_dict(torch.load('rosenbrock.pt'))

model.train()

# Test model
model.eval()
zt = np.zeros(np.shape(xx))
for xi in range(density):
	for yi in range(density):
		zt[xi,yi] = model(torch.tensor([[xx[xi,yi], yy[xi,yi]]]).type(dtype))[0,0].item()
min_si = np.unravel_index(np.argmin(zt, axis=None), zt.shape)
if DEBUG:
	ax2 = fig.add_subplot(122, projection='3d')
	ax2.plot_surface(xx,yy,zt, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	ax2.view_init(azim=0, elev=90)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.tight_layout()
	ax2.set_zticks([])
model.train()

## Controller
n_epochs = 100
loss_fn = torch.nn.MSELoss(reduction='sum')
inp = Variable(torch.tensor([[x0+0.0, y0+0.0]]).type(dtype), requires_grad=True)
optimizer = torch.optim.Adam([inp], lr=.2)
opt_steps_x = [x0+0.0]
opt_steps_y = [y0+0.0]
opt_steps_z = [f(x0,y0)]
for t in range(n_epochs):
	y0 = model(inp)
	loss = loss_fn(y0, torch.tensor([[y0.item()-0.1]]))
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	opt_steps_x.append(inp.data[0,0]+0.0)
	opt_steps_y.append(inp.data[0,1]+0.0)
	opt_steps_z.append(1.0)
	# opt_steps_z.append(f(inp.data[0,0]+0.0, inp.data[0,1]+0.0))

ax1.plot(opt_steps_x, opt_steps_y, opt_steps_z, 'k+', label='Steps', zorder=1e3)
if DEBUG:
	ax1.plot([opt_steps_x[-1]], [opt_steps_y[-1]], [opt_steps_z[-1]], 'r+', label='Steps', zorder=1e3)
	ax1.plot([xm],[ym],[1], 'g*', label='True Minimum', zorder=1e4, markersize=12)
ax1.plot([xx[min_si]],[yy[min_si]],[1], 'g*', label='Model Minimum', zorder=1e4, markersize=12)

if DEBUG:
	ax2.plot(opt_steps_x, opt_steps_y, opt_steps_z, 'k+', label='Steps', zorder=1e3)
	ax2.plot([opt_steps_x[-1]], [opt_steps_y[-1]], [opt_steps_z[-1]], 'r+', label='Steps', zorder=1e3)
	ax2.plot([xm],[ym],[1], 'g*', label='True', zorder=1e4)
	ax2.plot([xx[min_si]],[yy[min_si]],[1], 'm*', label='Model Minimum', zorder=1e4)

leg = ax1.legend(bbox_to_anchor=[0.5, 0.12], loc='lower center', ncol=3)
plt.show()