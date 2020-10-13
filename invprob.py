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

N = 1500		# simulation duration in steps
E = 100			# number of epochs
rho = 800		# learning rate
M = 1			# MPC look ahead

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

w_116 = (2*c*L*l*T+k*l*L*T*T) / (4*I+2*c*L*T+k*L*T*T)
w_146 = l*T*T/4/I
w_1411 = -L*T*T/4/I
w = w_146+w_1411*w_116
def Du(u_t, rho, x_t, r):
	return -rho*(x_t-r)*(w_146+w_1411*w_116)

def ref(t):
	return 1

time = np.arange(0, (N+M-1)*T, T)
u = np.zeros(len(time))
F = np.zeros(len(time))
x = np.zeros(len(time))
r = np.zeros(len(time))
J = np.zeros(len(time))
for t in range(len(time)-M+1):                # Real time
	r[t] = ref(time[t])
	dJdu = np.zeros(M)				# dJ[t+i]/du[t]
	dxdu = np.zeros(M)				# dx[t+i]/du[t]
	dudu = np.zeros(M)				# du[t+i]/du[t]
	for i in range(M):        # MPC looking ahead a few time steps
		# x_svec = []
		# u_svec = []
		# Du_svec = []
		
		u_guess = u[t+i-1]
		x_guess = x[t+i-1]
		for j in range(E):     # Training u for that time step
			Du_ij = Du(u_guess, rho, x_guess, ref(time[t+i-1]))
			u_guess += Du_ij
			x_guess = u2th(u_guess, u[t+i-1], u[t+i-2], x[t+i-1], x[t+i-2])
			
			# x_svec.append(x_guess)
			# Du_svec.append(Du_ij)
			# u_svec.append(u_guess)

		u[t+i] = u_guess
		F[t+i] = u2F(u[t+i], u[t+i-1], u[t+i-2], F[t+i-1], F[t+i-2])
		x[t+i] = x_guess
		dudu[i] = 1 if i==0 else dudu[i-1]-rho*w*dxdu[i-1]
		dxdu[i] = w if i==0 else w*dudu[i]
		x[t+i]
		time[t+i]
		dJdu[i] = (x[t+i]-ref(time[t+i]))*dxdu[i]

	# plt.figure()
	# plt.subplot(1,3,1)
	# plt.plot(range(N), u_svec)
	# plt.title('u')
	# plt.subplot(1,3,2)
	# plt.plot(range(N), x_svec)
	# plt.title('x_svec')
	# plt.subplot(1,3,3)
	# plt.plot(range(N), Du_svec)
	# plt.title('Du')
	# plt.show()

	u[t] = u[t-1]-rho*np.sum(dJdu)
	F[t] = u2F(u[t], u[t-1], u[t-2], F[t-1], F[t-2])
	x[t] = u2th(u[t], u[t-1], u[t-2], x[t-1], x[t-2])
	J[t] = (x[t]-r[t])**2

if M>1:
	time = np.delete(time, np.arange(len(time)-M+1,len(time)))
	u = np.delete(u, np.arange(len(u)-M+1,len(u)))
	F = np.delete(F, np.arange(len(F)-M+1,len(F)))
	x = np.delete(x, np.arange(len(x)-M+1,len(x)))
	r = np.delete(r, np.arange(len(r)-M+1,len(r)))

print(np.sum(J))

plt.figure()
plt.plot(time, u, label='u')
# plt.plot(time, F, label='F')
plt.plot(time, x, label='ξ')
plt.plot(time, r, label='r')
plt.xlabel('Time')
plt.legend()
plt.show()