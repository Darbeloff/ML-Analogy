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

N = 22       # number of epochs
rho = 600      # learning rate
TN = 1        # MPC look ahead

k = 1          # stiffness
T = 1e-2       #[s] sampling period
m = 1          #[kg] mass
c = 1          # damper
I = 1          # rotational inertia
l = 0.5        # distance to muscle attachement
L = 1          # length of arm

def u2F(u_t, u_tm1, u_tm2, f_tm1, f_tm2):
	return ((2*c*L*l*T+k*l*L*T*T)*u_t + 2*k*l*L*T*T*u_tm1 + (-2*c*L*l*T+k*l*L*T*T)*u_tm2 - (-8*I+2*k*L*T*T)*f_tm1 - (4*I-2*c*L*T+k*L*T*T)*f_tm2) / (4*I+2*c*L*T+k*L*T*T)

def uF2th(u_t, u_tm1, u_tm2, f_t, f_tm1, f_tm2, th_tm1, th_tm2):
	return l*T*T/4/I*(u_t+2*u_tm1+u_tm2) -L*T*T/4/I*(f_t+2*f_tm1+f_tm2) +2*th_tm1 -th_tm2

def end2end(u_t, u_tm1, u_tm2, f_tm1, f_tm2, th_tm1, th_tm2):
	f_t = u2F(u_t, u_tm1, u_tm2, f_tm1, f_tm2)
	th_t = uF2th(u_t, u_tm1, u_tm2, f_t, f_tm1, f_tm2, th_tm1, th_tm2)
	if np.isscalar(f_t):
		return (f_t, th_t)
	else:
		return np.concatenate((f_t[:,None],th_t[:,None]), 1)

def u2th(u_t, u_tm1, u_tm2, th_tm1, th_tm2):
	return (l*T*T*(u_t+2*u_tm1+u_tm2) -(-8*I+2*L*L*k*T*T)*th_tm1 -(4*I-2*L*L*c*T+L*L*k*T*T)*th_tm2) / (4*I+2*L*L*c*T+L*L*k*T*T)

def Du(u_t, rho, x_t, r):
	w_116 = (2*c*L*l*T+k*l*L*T*T) / (4*I+2*c*L*T+k*L*T*T)
	w_146 = l*T*T/4/I
	w_1411 = -L*T*T/4/I
	return -rho*(x_t-r)*(w_146+w_1411*w_116)

def ref(t):
	return 1

t_vec = np.arange(0, 10, T)
u_vec = [0,0]
F_vec = [0,0]
x_vec = [0,0]
e_vec = []
r_vec = []
for t in t_vec:                # Real time
	r_vec.append(ref(t))
	u_temp = l
	u_guess = u_vec[-1]
	for j in range(TN):        # MPC looking ahead a few time steps
		D_u = 0;
		for i in range(N):     # Training u for that time step
			x_guess = u2th(u_guess, u_vec[-1], u_vec[-2], x_vec[-1], x_vec[-2])
			Du_i = Du(u_guess, rho, x_guess, r_vec[-1])
			u_guess += Du_i
		u_vec.append(u_guess)
		F_vec.append(u2F(u_vec[-1], u_vec[-2], u_vec[-3], F_vec[-1], F_vec[-2]))
		x_vec.append(u2th(u_vec[-1], u_vec[-2], u_vec[-3], x_vec[-1], x_vec[-2]))
		D_u += Du_i

	del u_vec[-TN:]
	del F_vec[-TN:]
	del x_vec[-TN:]

	u_vec.append(u_vec[-1]+D_u)
	F_vec.append(u2F(u_vec[-1], u_vec[-2], u_vec[-3], F_vec[-1], F_vec[-2]))
	e_vec.append(x_vec[-1]-r_vec[-1])
	x_vec.append(u2th(u_vec[-1], u_vec[-2], u_vec[-3], x_vec[-1], x_vec[-2]))

plt.figure()
plt.plot(t_vec, u_vec[2:], label='u')
plt.plot(t_vec, F_vec[2:], label='F')
plt.plot(t_vec, x_vec[2:], label='x')
plt.plot(t_vec, e_vec, label='e')
plt.plot(t_vec, r_vec, label='r')
plt.axhline()
plt.xlabel('Time')
plt.title('Response to Unit Step Input')
plt.legend()
plt.show()

print(np.sum(np.absolute(e_vec)))