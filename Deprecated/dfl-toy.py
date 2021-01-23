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

def train_model(model, x, y):
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
				val_loss = val_loss_fn(y_val, yhat).item()
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
	plt.show()

	model.eval()
	return model

if __name__ == '__main__':
	torch.manual_seed(2)
	device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
	dtype = torch.FloatTensor

	k, b, m, dt = 1, 1, 1, 0.01
	N = 10000

	x = torch.rand(N, 2)
	u = torch.rand(N, 1)

	Fs = k*x[:,0][:,None]**3
	Fd = b*x[:,1][:,None]**3

	# Continuous-Time state-space matrices
	A_x = torch.tensor([[0,1],[0,0]])
	A_eta = torch.tensor([[0,0],[-1/m,-1/m]])
	B_x = torch.tensor([[0],[1/m]])

	# Convert state-space matrices to discrete-time
	A_x = dt*A_x+torch.tensor([[1,0],[0,1]])
	A_eta = dt*A_eta
	B_x = dt*B_x

	eta = torch.cat((Fs, Fd), 1)
	x_tp1 = torch.transpose(
		   torch.matmul(A_x, torch.transpose(x,0,1))
		 + torch.matmul(A_eta, torch.transpose(eta,0,1))
		 + torch.matmul(B_x, torch.transpose(u,0,1))
		,0,1)
	etad = torch.cat((3*k*x[:,0][:,None]**2, 3*b*x[:,1][:,None]**2), 1)
	eta_tp1 = dt*etad+eta
	xs = torch.cat((x, eta), 1)
	xs_tp1 = torch.cat((x_tp1, eta_tp1), 1)

	tilde_f = torch.nn.Linear(3, 2).to(device)
	tilde_g = torch.nn.Linear(5, 4).to(device)

	tilde_f = train_model(tilde_f, torch.cat((x, u), 1), x_tp1)
	torch.save(tilde_f.state_dict(), 'tilde_f.pt')

	tilde_g = train_model(tilde_g, torch.cat((xs, u), 1), xs_tp1)
	torch.save(tilde_g.state_dict(), 'tilde_g.pt')