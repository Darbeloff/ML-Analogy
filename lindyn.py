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
S = 10			# number of seconds to simulate
rho = 25		# learning rate
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
u = np.zeros((len(time),3))
xi = np.zeros((len(time),3))
r = np.zeros(len(time))
e = np.zeros((len(time),3))
J = np.zeros((len(time),3))
for t in range(len(time)-3):
	r[t] = ref(time[t])
	for i in range(3):
		J[t,i] = 0.5*(xi[t,i]-r[t])**2
		if t==0:
			u_tm1, u_tm2, xi_tm1 = 0, 0, 0
		elif t==1:
			u_tm1, u_tm2, xi_tm1 = u[t-1,i], 0, xi[t-1,i]
		else:
			u_tm1, u_tm2, xi_tm1 = u[t-1,i], u[t-2,i], xi[t-1,i]

		u_g = 1 if t==0 else u_tm1

		# u_g_j = np.zeros(M+1)
		# u_g_j[0] = u_g
		# Du_j = np.zeros(M+1)
		# for j in np.arange(1,M+1):
		# 	Du_j[j] = -rho*(u2x(u_g_j[j-1], u_tm1, u_tm2, xi[t], xi_tm1)-ref(time[t+1]))*dfdut
		# 	u_g_j[j] = u_g_j[j-1]+Du_j[j]
		# Du = np.sum(Du_j)

		xi_tp1 = u2x(u_g, u_tm1, u_tm2, xi[t,i], xi_tm1)
		dxitp1_dut = dfdut
		dJt_dut = (xi_tp1-ref(time[t+1]))*dxitp1_dut

		xi_tp2 = u2x(u_g, u_g, u_tm1, xi_tp1, xi[t,i])
		dxitp2_dut = dfdutm1+dfdxit*dfdut+dfdut
		dJtp1_dut = (xi_tp2-ref(time[t+2]))*dxitp2_dut

		xi_tp3 = u2x(u_g, u_g, u_g, xi_tp2, xi_tp1)
		dxitp3_dut = dfdut+dfdutm1+dfdutm2+dfdxit*dxitp2_dut+dfdxitm1*dfdut
		dJtp2_dut = (xi_tp3-ref(time[t+3]))*dxitp3_dut

		dJdu = dJt_dut
		if i>0: dJdu+= dJtp1_dut
		if i>1: dJdu+= dJtp2_dut
		Du = -rho*dJdu

		u[t,i] = u_g+Du
		xi[t+1,i] = u2x(u[t,i], u_tm1, u_tm2, xi[t,i], xi_tm1)

print(np.sum(J[:,2]))

plt.figure()
LW = 3
plt.plot(time[:-3], u[:-3,0],  label='u, N=1', linestyle=':',  color='blue', linewidth=LW)
plt.plot(time[:-3], u[:-3,1],  label='u, N=2', linestyle='--', color='blue', linewidth=LW)
plt.plot(time[:-3], u[:-3,2],  label='u, N=3', linestyle='-',  color='blue', linewidth=LW)
plt.plot(time[:-3], xi[:-3,0], label='ξ, N=1', linestyle=':',  color='orange', linewidth=LW)
plt.plot(time[:-3], xi[:-3,1], label='ξ, N=2', linestyle='--', color='orange', linewidth=LW)
plt.plot(time[:-3], xi[:-3,2], label='ξ, N=3', linestyle='-',  color='orange', linewidth=LW)
plt.plot(time[:-3], r[:-3], label='r', color='green', linewidth=LW)
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.show()