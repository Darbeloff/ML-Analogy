import numpy as np
from casadi import *
import do_mpc

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

x0 = np.array([0,0]).reshape(-1,1)

simulator.x0 = x0

graphics = do_mpc.graphics.Graphics()

fig, ax = plt.subplots(2, sharex=True)

graphics.add_line(var_type='_x', var_name='pos', axis=ax[0])
# Fully customizable:
ax[0].set_ylabel('x')

graphics.add_line(var_type='_u', var_name='force', axis=ax[1])
# Fully customizable:
ax[0].set_ylabel('u')
# ax[0].set_ylim(...)

for k in range(10000):
	u0 = mpc.make_step(x0)
	y_next = simulator.make_step(u0)
	x0 = estimator.make_step(y_next)

	graphics.reset_axes()
	graphics.plot_results(mpc.data, linewidth=3)
	graphics.plot_predictions(mpc.data, linestyle='--', linewidth=1)
	plt.show()
	input('next step')

graphics.plot_results(mpc.data)