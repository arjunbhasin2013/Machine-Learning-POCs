import compute_prediction as cp
import numpy as np

def predictGD_(X, weights):
	if X.shape[1] == weights.shape[0] - 1:
		intercept = np.ones((X.shape[0],1))
		X = np.hstack((intercept, X))
	return cp.gd_prediction_(X, weights)
