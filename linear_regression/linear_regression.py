import numpy as np


class LinearRegression:
    # input learning_rate, iterators, weights, bias
    def __init__(self, lr = 0.001, iters = 1000) -> None:
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None

    # fit datasets to train
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        self.X_train = X
        self.y_train = y

        for _ in range(self.iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted