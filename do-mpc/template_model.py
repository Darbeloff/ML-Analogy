def template_model():
	k, b, m = 1, 1, 1

	# Obtain an instance of the do-mpc model class
	# and select time discretization:
	model_type = 'continuous' # either 'discrete' or 'continuous'
	model = do_mpc.model.Model(model_type)

	# Introduce new states, inputs and other variables to the model, e.g.:
	pos = model.set_variable(var_type='_x', var_name='pos', shape=(1,1))
	dpos = model.set_variable(var_type='_x', var_name='dpos', shape=(1,1))
	u = model.set_variable(var_type='_u', var_name='force', shape=(1,1))
	ddpos = model.set_variable('_z', 'ddpos')

	model.set_rhs('pos', dpos)
	model.set_rhs('dpos', ddpos)
	
	model.set_alg('dynamics', m*ddpos+b*dpos**3+k*pos**3-u)

	# Setup model:
	model.setup()

	return model