import numpy as np
import compute_prediction as cp

def cost_(X, y, weights):
	'''Compute the cost J(w)
	
	Args : X, y (numpy array, data set)
	       weights
	       
	Returns : float
	'''
	
	predictions = cp.gd_prediction_(X, weights)
	cost = np.mean(-y * np.log(predictions) - (1-y) * np.log(1-predictions))
	return cost
