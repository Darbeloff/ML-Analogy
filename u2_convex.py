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

		if t>100 and validation_losses[-2]<=validation_losses[-1]:
			break

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

def lazyOpt(fn, lwrBnd, uprBnd, disc, intMask):
	n_vars = len(disc)

	step = []
	for v in range(n_vars):
		step.append((uprBnd[v]-lwrBnd[v])/disc[v])
	step = np.asarray(step)

	J = np.zeros(disc)
	for i in range(np.prod(disc)):
		idx = np.unravel_index(i,disc)
		inp = (np.asarray(lwrBnd).copy()+idx*step).tolist()
		for v in range(len(inp)):
			if intMask[v]:
				inp[v] = int(inp[v])
		J[idx] = controller(inp)

	mindx = np.asarray(np.unravel_index(np.argmin(J), disc))
	mindx_m1 = [max(0,v-1) for v in mindx]
	mindx_p1 = [min(uprBnd[v], mindx[v]+1) for v in range(len(mindx))]
	inp = np.asarray(lwrBnd).copy()
	inp_opt = inp+mindx*step
	inp_lwr = inp+mindx_m1*step
	inp_upr = inp+mindx_p1*step

	print('Optimal params: ',inp_opt)
	print('Between: ', inp_lwr, ' and ', inp_upr)

if __name__ == '__main__':
	torch.manual_seed(3) #5
	device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
	dtype = torch.FloatTensor

	k, b, m, dt, T = 1, 1, 1, 0.1, 200
	M = 10000

	x = (torch.rand(M, 2).type(dtype)-0.5)*2
	u = (torch.rand(M, 1).type(dtype)-0.5)*2

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

	def f_exact(x,u):
		return f(x, eta_fn(x,u), u)

	def f(x,eta,u):
		return torch.transpose(
			   torch.matmul(A_x,   torch.transpose(x,  0,1))
			 + torch.matmul(A_eta, torch.transpose(eta,0,1))
			 + torch.matmul(B_x,   torch.transpose(u**2,  0,1))
			,0,1)

	def h(x,eta,u):
		return dt*torch.cat((3*k*x[:,0][:,None]**2, 3*b*x[:,1][:,None]**2), 1)+eta

	# Initialize NN model
	H = 70
	# tilde_h = torch.nn.Sequential(
	# 			torch.nn.Linear(5, H),
	# 			torch.nn.ReLU(),
	# 			torch.nn.ReLU(),
	# 			torch.nn.Linear(H, 2),
	# 		).to(device)	# (x_t, eta_t, u_t) -> (eta_tp1)
	tilde_h = torch.nn.Linear(5, 2, bias=False)

	# Generate model
	eta = eta_fn(x,u)
	x_tp1 = f(x,eta,u)
	eta_tp1 = eta_fn(x_tp1,u)
	# eta_tp1 = h(x,eta,u)
	# tilde_h = train_model(tilde_h, torch.cat((x, eta, u), 1), eta_tp1)
	# torch.save(tilde_h.state_dict(), 'tilde_h.pt')

	# Load model
	tilde_h.load_state_dict(torch.load('tilde_h.pt'))

	# Compare H
	# etaobxi = torch.zeros((2,5))
	# xixi = torch.zeros((5,5))
	# for i in range(M):
	# 	etao = eta_tp1[i,:][:,None]
	# 	xi = torch.cat((x[i,:][:,None], eta[i,:][:,None], u[i][:,None]), 0)
	# 	etaobxi+= torch.matmul(etao, torch.transpose(xi,0,1))
	# 	xixi+= torch.matmul(xi, torch.transpose(xi,0,1))
	# etaobxi/= M
	# xixi/= M
	# H = torch.matmul(etaobxi, torch.inverse(xixi))
	# print('H=',H)
	# print('~H=',[h for h in tilde_h.parameters()])

	## Simulate model system
	'''
	x_vec = [0]
	x_app = [0]
	x_sim = [0]
	Fs_vec = [0]
	Fs_app = [0]
	Fs_sim = [0]
	Fd_vec = [0]
	Fd_app = [0]
	Fd_sim = [0]
	x_t = torch.tensor([[0,0]]).type(dtype)
	u_t = torch.tensor([[1]]).type(dtype)
	eta_t = torch.tensor([[0,0]]).type(dtype)
	eta_t_app = torch.tensor([[0,0]]).type(dtype)
	eta_t_sim = torch.tensor([[0,0]]).type(dtype)
	for t in range(T):
		# Forward propagate
		eta_tp1_app = h(x_t, eta_t, u_t)
		eta_tp1_sim = tilde_h(torch.cat((x_t, eta_t, u_t), 1))
		x_tp1 = f(x_t, eta_t, u_t)
		x_tp1_app = f(x_t, eta_t_app, u_t)
		x_tp1_sim = f(x_t, eta_t_sim, u_t)
		u_tp1 = torch.tensor([[1]]).type(dtype)
		eta_tp1 = eta_fn(x_tp1, u_tp1)

		# Record
		x_vec.append(x_tp1[0,0].item())
		x_app.append(x_tp1_app[0,0].item())
		x_sim.append(x_tp1_sim[0,0].item())
		Fs_vec.append(eta_tp1[0,0].item())
		Fd_vec.append(eta_tp1[0,1].item())
		Fs_app.append(eta_tp1_app[0,0].item())
		Fd_app.append(eta_tp1_app[0,1].item())
		Fs_sim.append(eta_tp1_sim[0,0].item())
		Fd_sim.append(eta_tp1_sim[0,1].item())

		# Iterate
		eta_t_app = eta_tp1_app.detach().clone()
		eta_t_sim = eta_tp1_sim.detach().clone()
		x_t = x_tp1.detach().clone()
		u_t = u_tp1.detach().clone()
		eta_t = eta_tp1.detach().clone()
	# Illustrate
	plt.figure()
	plt.subplot(1,3,1)
	plt.plot(range(T+1), x_vec, label='x')
	plt.plot(range(T+1), x_app, '*', label='x_app')
	plt.plot(range(T+1), x_sim, '+', label='x_sim')
	plt.legend()
	plt.subplot(1,3,2)
	plt.plot(range(T+1), Fs_vec, label='Fs')
	plt.plot(range(T+1), Fs_app, '*', label='Fs_app')
	plt.plot(range(T+1), Fs_sim, '+', label='Fs_sim')
	plt.legend()
	plt.subplot(1,3,3)
	plt.plot(range(T+1), Fd_vec, label='Fd')
	plt.plot(range(T+1), Fd_app, '*', label='Fd_app')
	plt.plot(range(T+1), Fd_sim, '+', label='Fd_sim')
	plt.legend()
	plt.show()
	'''

	# Controller
	def controller():
		# Q_x22, N, rho = inp
		x_vec = []
		u_vec = []
		r_vec = []
		x_t = torch.tensor([[0.5,0.5]]).type(dtype)
		u_t = Variable(torch.tensor([[-0.1]]).type(dtype), requires_grad=True)
		ref = lambda t : torch.tensor([[0,0]]).type(dtype)
		Q = torch.tensor([[1,0],[0,1]]).type(dtype)
		R = torch.tensor([[0]]).type(dtype)
		loss_fn = lambda x,r,u : (torch.matmul(r-x,torch.matmul(Q,torch.transpose(r-x,0,1))) + torch.matmul(u,torch.matmul(R,torch.transpose(u,0,1))))[0,0]
		u_poss = np.linspace(-0.6,0.6,50)
		tt,uu = np.meshgrid(range(T), u_poss, indexing='ij')
		J_surf = np.zeros(np.shape(tt))
		N = 20
		rho = 0.1
		optimizer = torch.optim.Adam([u_t], lr=rho)
		total_Cost = 0
		for t in range(T):
			for ui in range(len(u_poss)):
				J = torch.tensor(0).type(dtype)
				x_t_this = x_t.detach().clone()
				x_tpi = [x_t_this]
				u_this = torch.tensor([[u_poss[ui]]]).type(dtype)
				eta_tpi = [eta_fn(x_t_this, u_this)]
				for i in range(1,N+1):
					eta_tpi.append(tilde_h(torch.cat((x_tpi[-1], eta_tpi[-1], u_this), 1)))
					# x_tpi.append(f(x_tpi[-1], eta_tpi[-2], u_this))
					x_tpi.append(f_exact(x_tpi[-1], u_this))
					J+= loss_fn(x_tpi[-1], ref(t), u_this)
				# breakpoint()
				J_surf[t,ui] = J.item()

			J = torch.tensor(0).type(dtype)
			x_tpi = [x_t]
			eta_tpi = [eta_fn(x_t, u_t)]
			for i in range(1,N+1):
				eta_tpi.append(tilde_h(torch.cat((x_tpi[-1], eta_tpi[-1], u_t), 1)))
				# x_tpi.append(f(x_tpi[-1], eta_tpi[-2], u_t))
				x_tpi.append(f_exact(x_tpi[-1], u_t))
				J+= loss_fn(x_tpi[-1], ref(t), u_t)
			optimizer.zero_grad()
			J.backward()
			optimizer.step()
			u_vec.append(u_t[0,0].item())
			x_t = f_exact(x_t, u_t)
			x_t = x_t.detach().clone()
			x_vec.append(x_t[0,0].item())
			r_vec.append(ref(t)[0,0].item())
			total_Cost+= (x_vec[-1]-r_vec[-1])**2

			# print(100*t/T)

		# plt.figure()
		# plt.plot(range(T), x_vec, label='x')
		# plt.plot(range(T), r_vec, label='r')
		# plt.plot(range(T), u_vec, label='u')
		# plt.legend()
		# plt.xlabel('Time')
		# plt.show()

		fig = plt.figure()
		ax1 = fig.add_subplot(111, projection='3d')
		ax1.plot_surface(tt,uu,np.log(J_surf), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
		plt.plot(range(T), x_vec, label='x', zorder=10)
		plt.plot(range(T), r_vec, label='r', zorder=11)
		plt.plot(range(T), u_vec, label='u', zorder=12)
		ax1.view_init(azim=-90, elev=90)
		plt.legend(loc='lower center', ncol=3)
		plt.xlabel('Time')
		plt.tight_layout()
		ax1.set_zticks([])
		plt.show()

		return total_Cost

	print(controller())

	# lazyOpt(controller, (0,8,0.001), (0,30,.004), (4,4,4), (False, True, False))