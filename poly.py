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
N, H = 140, 256

f = lambda th: th**4+th**3-2*th**2-3*th

x = (torch.rand(N)-0.5)*4
y = f(x)

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

	train_loader = DataLoader(dataset=train_dataset, batch_size=D_in)
	val_loader = DataLoader(dataset=val_dataset, batch_size=D_in)

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

		print(f"[{t+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

		if validation_loss<1e-2:
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

model = train_model(x,y)
model.train()
plt.figure()
plt.plot(x,y,'k.')

## Test model
# x_test = np.linspace(-2,2,40)
# y_test = []
# for x in x_test:
# 	y_test.append(model(torch.tensor([[x]]).float()))
# 	plt.plot(x,y_test[-1].item(),'r+')
# plt.show()
# exit(0)

## Controller
n_epochs = 100
loss_fn = torch.nn.MSELoss(reduction='sum')
x0 = -0.9
inp = Variable(torch.tensor([[x0+0.0]]).type(dtype), requires_grad=True)
# gradient_mask = torch.zeros(1,1)
# gradient_mask[0,0] = 1.0
# inp.register_hook(lambda grad: grad.mul_(gradient_mask))
optimizer = torch.optim.Adam([inp], lr=.02)
for t in range(n_epochs):
	inp_cpy = inp.detach().clone()
	y0 = model(inp)
	loss = loss_fn(y0, torch.tensor([[y0.item()-0.1]]))
	optimizer.zero_grad()
	loss.backward()
	print(inp.data[0,0]+0.0)
	optimizer.step()
	print(inp.data[0,0]+0.0)
	print('~~~')
	plt.plot(inp.data[0,0]+0.0, f(inp.data[0,0]+0.0), 'r+')

plt.show()
exit(0)

def controller(model, u_tm1, u_tm2, th_tm1, th_tm2, ref):
	DEBUG = False
	loss_fn = lambda x, u: (x-ref)**2+5e-6*u**2
	n_epochs = 100
	learning_rate = 1e-1
	u_t = u_tm1

	if DEBUG:
		u_vec = [u_t]
		l_vec = []
		x_vec = []
		x_tru = []

	inp = Variable(torch.tensor([[u_t, u_tm1, u_tm2, th_tm1, th_tm2]]).type(dtype), requires_grad=True)
	gradient_mask = torch.zeros(1,5)
	gradient_mask[0,0] = 1.0
	inp.register_hook(lambda grad: grad.mul_(gradient_mask))

	optimizer = torch.optim.Adam([inp], lr=learning_rate)
	for t in range(n_epochs):
		inp_cpy = inp.detach().clone()
		x_t = model(inp)
		# print(x_t, inp[0,0])
		loss = loss_fn(x_t, inp[0,0])
		optimizer.zero_grad()
		# print('here')
		loss.backward()
		optimizer.step()
		# print(inp_cpy, inp, '~~')
		# exit(0)

		if DEBUG:
			u_vec.append(inp.data[0,0]+0.0)
			l_vec.append(loss.data)
			x_vec.append(x_t.data)
			x_tru.append(u2th(u_vec[-1],u_tm1, u_tm2, th_tm1, th_tm2))

	if DEBUG:
		plt.figure()
		plt.plot(range(n_epochs+1), u_vec, label='u')
		plt.plot(range(n_epochs), l_vec, label='loss')
		plt.plot(range(n_epochs), x_vec, label='x')
		plt.plot(range(n_epochs), x_tru, label='x_tru')
		plt.legend()
		plt.xlabel('Epoch')
		plt.show()

	# print(inp.data[0,0]+0.0)
	# exit(0)
	
	return inp.data[0,0]+0.0

# Create random Tensors to hold inputs and outputs
x = 2*np.random.rand(N, 5) # [u[t], u[t-1], u[t-2], f[t-1], f[t-2]]
model = train_model(x, u2th(x[:,0], x[:,1], x[:,2], x[:,3], x[:,4]))

def reference(t):
	return 1 # Unit Step
	#return np.sin(t)

t_vec = np.arange(0, 10, T)
u_vec = [0,0]
x_vec = [0,0]
x_vec_sim = []
for t in t_vec:
	u_vec.append(reference(t))
	x_vec_sim.append( model(torch.tensor([[u_vec[-1], u_vec[-2], u_vec[-3], x_vec[-1], x_vec[-2]]]).float()) )
	x_vec.append(u2th(u_vec[-1], u_vec[-2], u_vec[-3], x_vec[-1], x_vec[-2]))
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
x_vec_sim = []
l_vec = []
for t in t_vec:
	ref = reference(t)
	u_vec.append(controller(model, u_vec[-1], u_vec[-2], x_vec[-1], x_vec[-2], ref))
	x_vec_sim.append( model(torch.tensor([[u_vec[-1], u_vec[-2], u_vec[-3], x_vec[-1], x_vec[-2]]]).float()) )
	x_vec.append(u2th(u_vec[-1], u_vec[-2], u_vec[-3], x_vec[-1], x_vec[-2]))
	l_vec.append((x_vec[-1]-ref)**2+0.001*u_vec[-1]**2)
plt.figure()
plt.plot(t_vec, u_vec[2:], label='u')
plt.plot(t_vec, x_vec[2:], label='x')
plt.plot(t_vec, x_vec_sim, label='x_sim')
plt.plot(t_vec, l_vec, label='loss')
plt.xlabel('Time')
plt.title('Response to Unit Step Input')
plt.legend()
plt.show()