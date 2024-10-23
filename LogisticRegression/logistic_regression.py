import numpy as np


class LogisticRegression:
    def __init__(self, lr = 0.0001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # fit datasets to train
    def fit(self, X, y):
        n_samples, n_feature = X.shape
        self.weights = np.zeros(n_feature)
        self.bias = 0

        # self.X_train = X
        # self.y_train = y

        # gradient descent
        for _ in range(n_samples):
            # linear_model = f(x) = wx +b
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # dw, db
            dw = (1 / n_samples)* np.dot(X.T,(y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        #  Cost function 


    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_clf = [ 1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_clf

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))