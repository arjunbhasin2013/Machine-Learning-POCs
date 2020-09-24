import math
import numpy as np
import compute_prediction as cp
import train_file as tf
import predict as pr

if __name__ == "__main__":
	X_train = np.array([[6,7],[2,4],[3,6],[4,7],[1,6],[5,2],[2,0],[6,3],[4,1],[7,2]])
	y_train = np.array([0,0,0,0,0,1,1,1,1,1])

	weights = tf.train_logistic_regression_(X_train, y_train, max_iter = 1000, learning_rate = 0.1, fit_intercept = True)

	X_test = np.array([[7,4],[1,3],[3,1],[4,5]])
	predictions = pr.predictGD_(X_test, weights)

	print(predictions)
