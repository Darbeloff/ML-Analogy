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

def DFL(x_t, x_tm1, eta_fn, u_tm1):
	# Compute augmented state
	eta_tm1 = eta_fn(x_tm1)
	eta_t   = eta_fn(x_t  )

	# Dummy input
	u_t = torch.zeros(u_tm1.size())

	# Assemble xi
	xi_tm1 = torch.cat((x_tm1, eta_tm1, u_tm1), 0)
	xi_t   = torch.cat((x_t  , eta_t  , u_t  ), 0)

	# Linear regression to compute A and H
	xi_pinv = torch.pinverse(xi_tm1)
	A = torch.matmul(  x_t, xi_pinv)
	H = torch.matmul(eta_t, xi_pinv)

	return A, H, eta_t, xi_tm1

def train_model(model, x, y, title=None):
	# Reshape x and y to be vector of tensors
	x = torch.transpose(x,0,1)
	y = torch.transpose(y,0,1)

	dataset = TensorDataset(x, y)

	N_train = int(3*len(y)/5)
	train_dataset, val_dataset = random_split(dataset, [N_train,len(y)-N_train])

	train_loader = DataLoader(dataset=train_dataset, batch_size=50)
	val_loader   = DataLoader(dataset=val_dataset  , batch_size=50)

	loss_fn = torch.nn.MSELoss(reduction='sum')

	# Use the optim package to define an Optimizer that will update the weights of
	# the model for us. Here we will use Adam; the optim package contains many other
	# optimization algorithms. The first argument to the Adam constructor tells the
	# optimizer which Tensors it should update.
	learning_rate = .0001
	n_epochs = 1000
	training_losses = []
	validation_losses = []
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	def step(x_batch, y_batch, model, loss_fn):
		# Send data to GPU if applicable
		x_batch = x_batch.to(device)
		y_batch = y_batch.to(device)

		# Reshape data for linear algebra
		x_batch = torch.transpose(x_batch, 0,1)
		y_batch = torch.transpose(y_batch, 0,1)

		# Label input data
		x_tm1 = x_batch[:2,:]
		u_tm1 = x_batch[2,:][None,:]
		x_t   = y_batch

		A, H, eta_t, xi_tm1 = DFL(x_t, x_tm1, model, u_tm1)

		# Estimate x and eta using DFL model
		x_hat   = torch.matmul(A,xi_tm1)
		eta_hat = torch.matmul(H,xi_tm1)

		# Return 
		return loss_fn(x_t, x_hat) + loss_fn(eta_t, eta_hat)

	for t in range(n_epochs):
		batch_losses = []

		with torch.no_grad():
			val_losses = []
			for x_val, y_val in val_loader:
				val_loss = step(x_val, y_val, model, loss_fn).item()
				val_losses.append(val_loss)
			validation_loss = np.mean(val_losses)
			validation_losses.append(validation_loss)

		# if t>100 and validation_losses[-2]<=validation_losses[-1]:
		# 	break

		for x_batch, y_batch in train_loader:
			loss = step(x_batch, y_batch, model, loss_fn)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			batch_losses.append(loss.item())
		training_loss = np.mean(batch_losses)
		training_losses.append(training_loss)

		print(f"[{t+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
	
	plt.figure()
	plt.semilogy(range(len(training_losses)), training_losses, label='Training Loss')
	plt.semilogy(range(len(validation_losses)), validation_losses, label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	if title is not None:
		plt.title(title)
	plt.show()

	model.eval()
	return model

# def lazyOpt(fn, lwrBnd, uprBnd, disc, intMask):
# 	n_vars = len(disc)

# 	step = []
# 	for v in range(n_vars):
# 		step.append((uprBnd[v]-lwrBnd[v])/disc[v])
# 	step = np.asarray(step)

# 	J = np.zeros(disc)
# 	for i in range(np.prod(disc)):
# 		idx = np.unravel_index(i,disc)
# 		inp = (np.asarray(lwrBnd).copy()+idx*step).tolist()
# 		for v in range(len(inp)):
# 			if intMask[v]:
# 				inp[v] = int(inp[v])
# 		J[idx] = controller(inp)

# 	mindx = np.asarray(np.unravel_index(np.argmin(J), disc))
# 	mindx_m1 = [max(0,v-1) for v in mindx]
# 	mindx_p1 = [min(uprBnd[v], mindx[v]+1) for v in range(len(mindx))]
# 	inp = np.asarray(lwrBnd).copy()
# 	inp_opt = inp+mindx*step
# 	inp_lwr = inp+mindx_m1*step
# 	inp_upr = inp+mindx_p1*step

# 	print('Optimal params: ',inp_opt)
# 	print('Between: ', inp_lwr, ' and ', inp_upr)

def randu(m,n,a,b):
	return (b-a)*torch.rand(m,n)+a

class Transpose(torch.nn.Module):
	def __init__(self):
		super(Transpose, self).__init__()

	def forward(self, x):
		return torch.transpose(x, 0,1)

if __name__ == '__main__':
	# Options
	torch.manual_seed(3) #5
	device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
	dtype = torch.FloatTensor

	RETRAIN = False
	# RETRAIN = True

	# Parameters
	k, b, m, dt, T = 1, 1, 1, 0.1, 400
	M = 10000
	H, leta = 256, 2

	# Define state transition function
	def f(x,u):
		# Continuous-Time state-space matrices
		A_x = torch.tensor([[0,1],[0,0]]).type(dtype)
		A_eta = torch.tensor([[0,0],[-1/m,-1/m]]).type(dtype)
		B_x = torch.tensor([[0],[1/m]]).type(dtype)

		# Convert state-space matrices to discrete-time
		A_x = dt*A_x+torch.tensor([[1,0],[0,1]]).type(dtype)
		A_eta = dt*A_eta
		B_x = dt*B_x

		# Nonlinear elements
		Fs = k*x[0,:][None,:]**3
		Fd = b*x[1,:][None,:]**3
		eta = torch.cat((Fs, Fd), 0)

		return torch.matmul(A_x  ,   x) + \
			   torch.matmul(A_eta, eta) + \
			   torch.matmul(B_x  ,   u)

	# Create training data
	x_tm1 = randu(2,M,-1,1).type(dtype)
	u_tm1 = randu(1,M,-1,1).type(dtype)
	x_t = f(x_tm1,u_tm1)

	# Initialize model
	g = torch.nn.Sequential(
		Transpose(),
		torch.nn.Linear(2,H),
		torch.nn.ReLU(),
		torch.nn.ReLU(),
		torch.nn.ReLU(),
		torch.nn.ReLU(),
		torch.nn.Linear(H,leta),
		Transpose() )

	# Train model
	if RETRAIN:
		g = train_model(g, torch.cat((x_tm1, u_tm1), 0), x_t)
		torch.save(g.state_dict(), 'g.pt')
	else:
		g.load_state_dict(torch.load('g.pt'))
	A, H, _, _ = DFL(x_t, x_tm1, g, u_tm1)

	# Simulate step response
	x0 = torch.tensor([[0],[0]]).type(dtype)
	x   = float('nan')*torch.ones(2,T).type(dtype)
	xs  = float('nan')*torch.ones(2,T).type(dtype)
	eta = float('nan')*torch.ones(2,T).type(dtype)
	u   =          0.5*torch.ones(1,T).type(dtype)
	x  [:,0] = torch.squeeze(  x0 )
	xs [:,0] = torch.squeeze(  x0 )
	eta[:,0] = torch.squeeze(g(x0))
	for t in range(1,T):
		x[:,t] = f(x[:,t-1][:,None], u[:,t-1][:,None]).squeeze()
		xi_tm1 = torch.cat((xs[:,t-1], eta[:,t-1], u[:,t-1]), 0)[:,None]
		xs[:,t] = torch.matmul(A,xi_tm1).squeeze()
		eta[:,t] = torch.matmul(H,xi_tm1).squeeze()

	# Illustrate
	plt.figure()
	plt.plot(range(T), x[0,:].detach().numpy(), label='x')
	plt.plot(range(T), u[0,:].detach().numpy(), label='u')
	plt.plot(range(T), xs[0,:].detach().numpy(), label='xs')
	plt.xlabel('Time')
	plt.legend()
	plt.show()