import numpy as np
from casadi import *
import do_mpc, pdb

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

x0 = np.array([0,0]).reshape(-1,1)

simulator.x0 = x0

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(1)

for g in [sim_graphics, mpc_graphics]:
	# Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
	g.add_line(var_type='_x', var_name='pos', axis=ax)
	# g.add_line(var_type='_u', var_name='force', axis=ax)

	# Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:

ax.set_ylabel('x')
ax.set_xlabel('time [s]')

# u0 = mpc.make_step(x0)
# sim_graphics.clear()
# mpc_graphics.plot_predictions()
# mpc_graphics.reset_axes()
# # Show the figure:
# plt.show()

for i in range(20):
	u0 = mpc.make_step(x0)
	x0 = simulator.make_step(u0)

# Plot predictions from t=0
mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()
plt.show()