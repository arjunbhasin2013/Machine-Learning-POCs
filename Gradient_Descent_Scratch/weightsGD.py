import compute_prediction as cp
import numpy as np

def update_weights_gd(X_train, y_train, weights, learning_rate):
	'''Update weights by one step
	
	Returns : numpy array, updated weights
	'''
	
	predictions = cp.gd_prediction_(X_train, weights)
	weights_delta = np.dot(X_train.T, y_train - predictions)
	m = y_train.shape[0]
	weights+=learning_rate/float(m) * weights_delta 
	return weights
