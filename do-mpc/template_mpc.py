import numpy as np
from casadi import *
import do_mpc

def template_mpc(model):
	# Obtain an instance of the do-mpc MPC class
	# and initiate it with the model:
	mpc = do_mpc.controller.MPC(model)

	# Set parameters:
	setup_mpc = {
		'n_horizon': 20,
		'n_robust': 0,
		'open_loop': 0,
		't_step': 0.1,
		'state_discretization': 'collocation',
		'collocation_type': 'radau',
		'collocation_deg': 3,
		'collocation_ni': 1,
		'store_full_solution': True,
		# Use MA27 linear solver in ipopt for faster calculations:
		'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
	}
	mpc.set_param(**setup_mpc)

	# Configure objective function:
	mterm = (model._x['pos'] - 1)**2    # Setpoint tracking
	lterm = (model._x['pos'] - 1)**2    # Setpoint tracking
	# mterm+= 0.1*model._x['dpos']**2    # Setpoint tracking
	# lterm+= 0.1*model._x['dpos']**2    # Setpoint tracking

	mpc.set_objective(mterm=mterm, lterm=lterm)
	mpc.set_rterm(force=0) # Scaling for quad. cost.

	mpc.setup()

	return mpc