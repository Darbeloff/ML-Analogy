# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 1000, 5, 256, 1
seq_length = 10

k = 1          # spring constant
T = 1e-1       #[s] sampling period
m = 1          #[kg] mass
c = 1          # damper
l = 0.5        # moment arm of muscle
L = 1          # moment arm of spring

def haptic_feedback(f_tm1, f_tm2, u_t, u_tm1, u_tm2):
	return (3*k*T*T*u_t + 6*k*T*T*u_tm1 + 3*k*T*T*u_tm2 - 2*l*(k*T*T-4*m*L)*f_tm1 - l*(4*m*L+k*T*T)*f_tm2) / (l*(4*m*L+k*T*T))

def controller(e):
	return k*e

def reference(t):
	return 1 # Unit Step
	#return np.sin(t)

class LSTM(nn.Module):
	def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size

		self.lstm = nn.LSTM(input_size, hidden_layer_size)

		self.linear = nn.Linear(hidden_layer_size, output_size)

		self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
							torch.zeros(1,1,self.hidden_layer_size))

	def forward(self, input_seq):
		lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
		predictions = self.linear(lstm_out.view(len(input_seq), -1))
		return predictions[-1]

'''
t_vec = np.arange(0,10,T)
f_vec = [0,0]
u_vec = [0,0]
for t in t_vec:
	u_vec.append(1)
	f_vec.append(haptic_feedback(f_vec[-1], f_vec[-2], u_vec[-1], u_vec[-2], u_vec[-3]))
f_vec.pop(0)
f_vec.pop(0)
u_vec.pop(0)
u_vec.pop(0)

plt.figure()
plt.plot(t_vec,f_vec)
plt.show()
exit(0)
'''

# Create random Tensors to hold inputs and outputs
u = np.cumsum(2*np.random.rand(seq_length, N)-1,0) # cumsum of unif rand (-1,1)
f = np.zeros((2,N))
for t in range(seq_length):
	f_next = np.zeros((1,N))
	for i in range(N):
		u_tm1 = 0 if t<1 else u[t-1,i]
		u_tm2 = 0 if t<2 else u[t-2,i]
		f_next[0,i] = haptic_feedback(f[-1,i],f[-2,i],u[t,i],u_tm1,u_tm2)
	f = np.concatenate((f,f_next),0)
f = np.delete(f,0,0)
f = np.delete(f,0,0)

train_inout_seq = []
val_inout_seq = []
for i in range(N):
	if i<0.8*N:
		train_inout_seq.append((torch.from_numpy(u[:,i]).float(),torch.tensor(f[-1,i]).float()))
	else:
		val_inout_seq.append((torch.from_numpy(u[:,i]).float(),torch.tensor(f[-1,i]).float()))

# Use the nn package to define our model and loss function.
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
train_losses = []
val_losses = []
n_epochs = 1000
for i in range(n_epochs):
	model.train()
	train_loss = 0
	for seq, labels in train_inout_seq:
		seq = seq.to(device)
		labels = labels.to(device)
		optimizer.zero_grad()
		model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
						torch.zeros(1, 1, model.hidden_layer_size))

		y_pred = model(seq)

		single_loss = loss_function(y_pred, labels)
		train_loss+=single_loss.item()
		single_loss.backward()
		optimizer.step()
	train_loss/= len(train_inout_seq)
	train_losses.append(train_loss)

	model.eval()
	val_loss = 0
	with torch.no_grad():
		for seq, labels in val_inout_seq:
			seq = seq.to(device)
			labels = labels.to(device)
			model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
						torch.zeros(1, 1, model.hidden_layer_size))

			y_pred = model(seq)

			single_loss = loss_function(y_pred, labels)
			val_loss+=single_loss.item()
	val_loss/= len(val_inout_seq)
	val_losses.append(val_loss)

	print(f'epoch: {i:3} train loss: {train_loss:10.8f} val loss: {val_loss:10.8f}')

	if i>40 and np.median(val_losses[-20:-1])>=np.median(val_losses[-40:-21]):
		break

model.eval()

plt.figure()
plt.semilogy(range(len(train_losses)), train_losses, label='Training Loss')
plt.semilogy(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

t_vec = np.arange(0, seq_length*T, T)
u_vec = [0,0]
f_vec = [0,0]
f_sim_vec = []
for t in t_vec:
	u_vec.append(1)

	f_vec.append(haptic_feedback(f_vec[-1], f_vec[-2], u_vec[-1], u_vec[-2], u_vec[-3]))

u_vec.pop(0)
u_vec.pop(0)
f_vec.pop(0)
f_vec.pop(0)

print(f_vec[-1],model(torch.tensor(u_vec).float()).item())