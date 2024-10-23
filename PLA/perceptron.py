import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000) -> None:
        """
        Initializes the Perceptron with a learning rate and number of iterations.

        Parameters:
        lr (float): Learning rate (default: 0.01)
        n_iters (int): Number of iterations for training (default: 1000)
        """
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func  # Activation function
        self.weights = None
        self.bias = None

    def _unit_step_func(self, x):
        """
        Unit step activation function.

        Parameters:
        x (float or np.ndarray): Input value(s)

        Returns:
        int or np.ndarray: 1 if x >= 0 else 0
        """
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """
        Trains the Perceptron on the given dataset.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Target vector of shape (n_samples,)
        """
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert y to binary values (0 or 1)
        y_ = np.array([1 if i > 0 else 0 for i in y])

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Calculate linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply activation function
                y_predicted = self._unit_step_func(linear_output)
                # Perceptron update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Parameters:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
        np.ndarray: Predicted class labels of shape (n_samples,)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._unit_step_func(linear_model)
        return y_predicted



'''
Explanation of the Code and Variables:
Perceptron Class:

__init__: Initializes the Perceptron with a learning rate (lr) and number of iterations (n_iters). Sets the activation function to a unit step function and initializes weights and bias to None.
_unit_step_func: A private method that implements the unit step function, which returns 1 if the input is greater than or equal to 0, otherwise returns 0.
fit: Trains the Perceptron using the input feature matrix X and target vector y. It initializes the weights and bias, then iterates over the dataset to update the weights and bias based on the Perceptron update rule.
predict: Predicts the class labels for the input data X using the trained weights and bias.
Variables:

lr: Learning rate for the weight updates.
n_iters: Number of iterations for training.
activation_func: The activation function used in the Perceptron (unit step function in this case).
weights: The weights for the features.
bias: The bias term.
X: Input feature matrix.
y: Target vector.
n_samples: Number of samples in the dataset.
n_features: Number of features in the dataset.
linear_output: The linear combination of the input features and weights plus the bias.
y_predicted: The predicted output after applying the activation function.
update: The value used to update the weights and bias based on the Perceptron update rule.
'''