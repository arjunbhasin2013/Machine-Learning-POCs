import sigmoid as sg
import numpy as np

def gd_prediction_(X, weights):
	'''
	Compute the prediction y_hat based on current weights
	
	Args: X(numpy array)
	      weights (numpy array)
	
	Returns : numpy array, y_hat of X under weights
	'''
	
	z = np.dot(X, weights)
	prediction = sg.compute_sigmoid_(z)
	return prediction
