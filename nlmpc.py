import numpy as np
import copy, time

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

def train_model(model, x, y, title=None):
	if len(np.shape(x))==1:
		x = x[:, None]
	if len(np.shape(y))==1:
		y = y[:, None]
	D_in = np.shape(x)[1]
	D_out = np.shape(y)[1]
	N = 50

	dataset = TensorDataset(x, y)

	N_train = int(3*len(y)/5)
	train_dataset, val_dataset = random_split(dataset, [N_train,len(y)-N_train])

	train_loader = DataLoader(dataset=train_dataset, batch_size=N)
	val_loader = DataLoader(dataset=val_dataset, batch_size=N)

	loss_fn = torch.nn.MSELoss(reduction='sum')
	val_loss_fn = lambda target, output: loss_fn(target[:,:2], output[:,:2])

	# Use the optim package to define an Optimizer that will update the weights of
	# the model for us. Here we will use Adam; the optim package contains many other
	# optimization algorithms. The first argument to the Adam constructor tells the
	# optimizer which Tensors it should update.
	learning_rate = .0001
	n_epochs = 10000
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

		if t>100 and validation_losses[-2]<=validation_losses[-1]:
			break
	
	plt.figure()
	plt.semilogy(range(len(training_losses)), training_losses, label='Training Loss')
	plt.semilogy(range(len(training_losses)), validation_losses, label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	if title is not None:
		plt.title(title)
	plt.show()

	model.eval()
	return model

if __name__ == '__main__':
	torch.manual_seed(2)
	device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
	dtype = torch.FloatTensor

	k, b, m, dt = 1, 1, 1, 0.1
	N = 10000

	x = torch.rand(N, 2).type(dtype)
	u = torch.rand(N, 1).type(dtype)

	# Continuous-Time state-space matrices
	A_x = torch.tensor([[0,1],[0,0]]).type(dtype)
	A_eta = torch.tensor([[0,0],[-1/m,-1/m]]).type(dtype)
	B_x = torch.tensor([[0],[1/m]]).type(dtype)

	# Convert state-space matrices to discrete-time
	A_x = dt*A_x+torch.tensor([[1,0],[0,1]]).type(dtype)
	A_eta = dt*A_eta
	B_x = dt*B_x

	# Define state transition function
	def eta_fn(x,u):
		Fs = k*x[:,0][:,None]**3
		Fd = b*x[:,1][:,None]**3
		return torch.cat((Fs, Fd), 1)

	def f(x,u,eta=None):
		if eta is None:
			eta = eta_fn(x,u)
		return torch.transpose(
			   torch.matmul(A_x, torch.transpose(x,0,1))
			 + torch.matmul(A_eta, torch.transpose(eta,0,1))
			 + torch.matmul(B_x, torch.transpose(u,0,1))
			,0,1)

	def h(x,u,eta):
		return torch.cat((3*k*x[:,0][:,None]**2, 3*b*x[:,1][:,None]**2), 1)

	# Initialize NN models
	tilde_f = torch.nn.Linear(3,2).to(device)	# (x_t, u_t) -> (x_tp1)
	tilde_g = torch.nn.Linear(5,2).to(device)	# (x_t, eta_t, u_t) -> (x_tp1)
	tilde_h = torch.nn.Linear(5,2).to(device)	# (x_t, eta_t, u_t) -> (eta_tp1)

	# Generate models
	# eta = eta_fn(x,u)
	# x_tp1 = f(x,u,eta)
	# eta_tp1 = h(x,u,eta)
	# tilde_f = train_model(tilde_f, torch.cat((x, u), 1), x_tp1)
	# torch.save(tilde_f.state_dict(), 'nlmpc_models/tilde_f.pt')
	# tilde_g = train_model(tilde_g, torch.cat((x, eta, u), 1), x_tp1)
	# torch.save(tilde_g.state_dict(), 'nlmpc_models/tilde_g.pt')
	# tilde_h = train_model(tilde_h, torch.cat((x, eta, u), 1), eta_tp1)
	# torch.save(tilde_h.state_dict(), 'nlmpc_models/tilde_h.pt')

	# Load models
	tilde_f.load_state_dict(torch.load('nlmpc_models/tilde_f.pt'))
	tilde_g.load_state_dict(torch.load('nlmpc_models/tilde_g.pt'))
	tilde_h.load_state_dict(torch.load('nlmpc_models/tilde_h.pt'))

	# Simulate model system
	# x_vec = [0]
	# x_sim = [0]
	# x_t = torch.tensor([[0,0]]).type(dtype)
	# T = 100
	# for t in range(T):
	# 	u = torch.tensor([[1]]).type(dtype)
	# 	eta_t = eta_fn(x_t,u)
	# 	# xs_t = tilde_g(torch.cat((x_t, eta_t, u), 1).type(dtype))
	# 	xs_t = tilde_f(torch.cat((x_t, u), 1).type(dtype))
	# 	x_t = f(x_t, u, eta_t)
	# 	x_vec.append(x_t[0,0].item())
	# 	x_sim.append(xs_t[0,0].item())
	# plt.figure()
	# plt.plot(range(T+1), x_vec, label='x')
	# plt.plot(range(T+1), x_sim, '*', label='x_sim')
	# plt.show()

	# Controller
	x_vec = [0]
	u_vec = []
	r_vec = []
	x_t = torch.tensor([[0,0]]).type(dtype)
	u_t = Variable(torch.tensor([[1]]).type(dtype), requires_grad=True)
	T = 300
	ref = lambda t : torch.tensor([[1,0]]).type(dtype)
	Q = torch.tensor([[1,0],[0,1]]).type(dtype)
	R = torch.tensor([[0]]).type(dtype)
	loss_fn = lambda x,r,u : (torch.matmul(r-x,torch.matmul(Q,torch.transpose(r-x,0,1))) + torch.matmul(u,torch.matmul(R,torch.transpose(u,0,1))))[0,0]
	N = 20
	rho = 0.1
	optimizer = torch.optim.Adam([u_t], lr=rho)
	for t in range(T):
		J = torch.tensor(0).type(dtype)
		x_tpi = [x_t]
		for i in range(1,N+1):
			eta_tpim1 = eta_fn(x_tpi[i-1], u_t)
			x_tpi.append(f(x_tpi[i-1], u_t, eta_tpim1))
			# x_tpi.append(tilde_g(torch.cat((x_tpi[i-1], eta_tpim1, u_t), 1)))
			# x_tpi.append(tilde_f(torch.cat((x_tpi[i-1], u_t), 1)))
			J+= loss_fn(x_tpi[-1], ref(t), u_t)
		optimizer.zero_grad()
		J.backward()
		optimizer.step()
		u_vec.append(u_t[0,0].item())
		x_t = f(x_t, u_t)
		x_t = x_t.detach().clone()
		x_vec.append(x_t[0,0].item())
		r_vec.append(ref(t)[0,0].item())

	plt.figure()
	plt.plot(range(T+1), x_vec, label='x')
	plt.plot(range(T), u_vec, label='u')
	plt.plot(range(T), r_vec, label='r')
	plt.legend()
	plt.xlabel('Time')
	plt.show()