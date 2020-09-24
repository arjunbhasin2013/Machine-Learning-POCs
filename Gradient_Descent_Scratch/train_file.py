import numpy as np
import cost_compute as cc
import weightsGD as wt

def train_logistic_regression_(X_train, y_train, max_iter, learning_rate, fit_intercept = False):
	''' Training a logistic regression model
	
	Args: X_train, y_train (numpy arrays, training data set)
	max_iter (int, number of iterations)
	learning rate(float)
	fit_intercept(bool, with an intercept w0 ot not)
	
	Returns : numpy array, learned weights
	'''
	
	if fit_intercept:
		intercept = np.ones((X_train.shape[0], 1))
		X_train = np.hstack((intercept, X_train))
	weights = np.zeros(X_train.shape[1])
	
	for iteration in range(max_iter):
		weights = wt.update_weights_gd(X_train, y_train, weights, learning_rate)
		
		#~ CHECK THE COST AT EVERY 100 ITERATIONS
		if iteration%100 == 0:
			print(cc.cost_(X_train, y_train, weights))
	
	return weights
