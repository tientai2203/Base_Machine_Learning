import numpy as np

class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000) -> None:
        """
        Initialize the SVM model with hyperparameters.
        
        Parameters:
        lr (float): Learning rate for gradient descent.
        lambda_param (float): Regularization parameter.
        n_iters (int): Number of iterations for training.
        """
        self.lr = lr  # Learning rate for gradient descent
        self.lambda_param = lambda_param  # Regularization parameter
        self.n_iters = n_iters  # Number of iterations for training
        self.weights = None  # Weights for the features
        self.bias = None  # Bias term

    def fit(self, X, y):
        """
        Train the SVM model using the given training data.
        
        Parameters:
        X (numpy.ndarray): Training data features, shape (n_samples, n_features).
        y (numpy.ndarray): Training data labels, shape (n_samples,).
        """
        # Convert labels to -1 and 1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent for n_iters iterations
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check the condition for updating weights and bias
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    # Update weights for the regularization term only
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    # Update weights and bias for misclassified samples
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        """
        Predict the labels for given data.
        
        Parameters:
        X (numpy.ndarray): Data features, shape (n_samples, n_features).
        
        Returns:
        numpy.ndarray: Predicted labels, shape (n_samples,).
        """
        # Compute the linear output and apply the sign function
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)


'''
Explanation of Variables:
self.lr: Learning rate for the gradient descent optimization.
self.lambda_param: Regularization parameter to prevent overfitting.
self.n_iters: Number of iterations for training the model.
self.weights: Coefficients for the features.
self.bias: Bias term.
X: Training data features, a 2D NumPy array with shape (n_samples, n_features).
y: Training data labels, a 1D NumPy array with shape (n_samples,).
y_: Modified labels, where values <= 0 are set to -1 and > 0 are set to 1.
n_samples: Number of samples in the training data.
n_features: Number of features in the training data.
condition: Condition to check if the sample is correctly classified.
linear_output: Result of the linear combination of weights and features minus the bias, used for making predictions.
'''