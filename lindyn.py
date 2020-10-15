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

N = 1		    # simulation duration in steps
S = 15			# number of seconds to simulate
rho = 80		# learning rate
M = 5			# MPC look ahead

k = 1          # stiffness
T = 1e-2       #[s] sampling period
m = 1          #[kg] mass
c = 1          # damper
I = 1          # rotational inertia
l = 0.5        # distance to muscle attachement
L = 1          # length of arm

dfdut = l*T*T / (4*I+2*L*L*c*T+L*L*k*T*T)
dfdutm1 = l*T*T*2 / (4*I+2*L*L*c*T+L*L*k*T*T)
dfdutm2 = l*T*T / (4*I+2*L*L*c*T+L*L*k*T*T)
dfdxit = -(-8*I+2*L*L*k*T*T) / (4*I+2*L*L*c*T+L*L*k*T*T)
dfdxitm1 = -(4*I-2*L*L*c*T+L*L*k*T*T) / (4*I+2*L*L*c*T+L*L*k*T*T)
def u2x(u_t, u_tm1, u_tm2, xi_t, xi_tm1):
	# return (l*T*T*(u_t+2*u_tm1+u_tm2) -(-8*I+2*L*L*k*T*T)*xi_t -(4*I-2*L*L*c*T+L*L*k*T*T)*xi_tm1) / (4*I+2*L*L*c*T+L*L*k*T*T)
	return dfdut*u_t + dfdutm1*u_tm1 + dfdutm2*u_tm2 + dfdxit*xi_t + dfdxitm1*xi_tm1

def ref(t):
	return 1

time = np.arange(0, S, T)
u = np.zeros(len(time))
xi = np.zeros(len(time))
r = np.zeros(len(time))
e = np.zeros(len(time))
J = np.zeros(len(time))
for t in range(len(time)-2):
	r[t] = ref(time[t])
	J[t] = 0.5*(xi[t]-r[t])**2
	if t==0:
		u_tm1, u_tm2, xi_tm1 = 0, 0, 0
	elif t==1:
		u_tm1, u_tm2, xi_tm1 = u[t-1], 0, xi[t-1]
	else:
		u_tm1, u_tm2, xi_tm1 = u[t-1], u[t-2], xi[t-1]

	u_g = 1 if t==0 else u_tm1

	# u_g_j = np.zeros(M+1)
	# u_g_j[0] = u_g
	# Du_j = np.zeros(M+1)
	# for j in np.arange(1,M+1):
	# 	Du_j[j] = -rho*(u2x(u_g_j[j-1], u_tm1, u_tm2, xi[t], xi_tm1)-ref(time[t+1]))*dfdut
	# 	u_g_j[j] = u_g_j[j-1]+Du_j[j]
	# Du = np.sum(Du_j)

	xi_tp1 = u2x(u_g, u_tm1, u_tm2, xi[t], xi_tm1)
	dxitp1_dut = dfdut
	dJt_dut = (xi_tp1-ref(time[t+1]))*dxitp1_dut

	xi_tp2 = u2x(u_g, u_g, u_tm1, xi_tp1, xi[t])
	dxitp2_dut = dfdutm1+dfdxit*dfdut+dfdut
	dJtp1_dut = (xi_tp2-ref(time[t+2]))*dxitp2_dut

	Du = -rho*(dJt_dut+dJtp1_dut)

	u[t] = u_g+Du
	xi[t+1] = u2x(u[t], u_tm1, u_tm2, xi[t], xi_tm1)

print(np.sum(J))

plt.figure()
plt.plot(time[:-2], u[:-2], label='u')
plt.plot(time[:-2], xi[:-2], label='ξ')
plt.plot(time[:-2], r[:-2], label='r')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()