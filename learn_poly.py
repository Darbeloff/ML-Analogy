# -*- coding: utf-8 -*-
import numpy as np
import torch, random
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
D_in, D_out, N, H = 1, 1, 50, 256

f = lambda th: th**4+th**3-2*th**2-3*th
x_samp = np.linspace(-2,2,100)
y_tru = f(x_samp)
y_mod = np.zeros(np.shape(x_samp))

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
rho_model = .002
rho_controller = .02
n_epochs = 3000
x0 = -1.8
inp = Variable(torch.tensor([[x0+0.0]]).type(dtype), requires_grad=True)
opt_model = torch.optim.Adam(model.parameters(), lr=rho_model)
opt_controller = torch.optim.Adam([inp], lr=rho_controller)
x_g = np.zeros(n_epochs)
x_g[0] = x0
x_g_t = torch.from_numpy(x_g).float()
y_g = np.zeros(n_epochs)
y_g[0] = f(x0)
y_g_t = torch.from_numpy(y_g).float()
plt.figure()
for t in range(n_epochs):
	# Prepare illustration
	plt.clf()
	plt.plot(x_samp,y_tru,'k-')

	i_batch = random.choices(np.arange(t+1), k=min(20, t+1))
	x_batch = x_g_t[i_batch].detach().clone()
	y_batch = y_g_t[i_batch].detach().clone()
	breakpoint()
	y_pred = model(x_batch)

	# Train Model
	loss = loss_fn(y_pred, y_batch)
	opt_model.zero_grad()
	opt_controller.zero_grad()
	loss.backward()
	opt_model.step()

	# Eval Model
	model.eval()
	for i in range(len(x_samp)):
		y_mod[i] = model(torch.tensor([[x_samp[i]]]).float())
	plt.plot(x_samp, y_mod, 'r-')
	model.train()

	# Run Controller
	y0 = model(inp)
	y_goal = torch.tensor([[y0.item()-0.1]])
	loss = loss_fn(y0, y_goal)
	opt_model.zero_grad()
	opt_controller.zero_grad()
	loss.backward()
	opt_controller.step()
	x_g.append(inp[0,0].item())
	y_g.append(f(x_g[-1]))

	# Illustrate Controller
	plt.plot(x_g,y_g,'b+')

	plt.show(block=False)
	plt.pause(0.5)