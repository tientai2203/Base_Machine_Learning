import numpy as np

class NaiveBayes:
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        # Number of samples and number of features
        n_samples, n_features = X.shape
        # Unique class labels
        self._classes = np.unique(y)
        # Number of unique classes
        n_classes = len(self._classes)

        # Initialize mean, variance, and prior probability arrays
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            # Extract samples belonging to class c
            X_c = X[c == y]
            # Calculate mean for each feature for class c
            self._mean[idx, :] = X_c.mean(axis=0)
            # Calculate variance for each feature for class c
            self._var[idx, :] = X_c.var(axis=0)
            # Calculate prior probability for class c
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        # Predict class labels for each sample in X
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        # List to store posterior probabilities for each class
        posteriors = []

        for idx, c in enumerate(self._classes):
            # Log of prior probability for class c
            prior = np.log(self._priors[idx])
            # Sum of log of class conditional probabilities for class c
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            # Posterior probability for class c (log scale)
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # Return the class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        # Mean and variance for the given class index
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        # Calculate the probability density function (Gaussian distribution)
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


'''
Explanation of Key Variables and Methods:
X: Feature matrix of shape (n_samples, n_features).
y: Target labels of shape (n_samples,).
n_samples: Number of samples in the dataset.
n_features: Number of features in the dataset.
self._classes: Array of unique class labels.
n_classes: Number of unique classes.
self._mean: Array to store the mean of each feature for each class.
self._var: Array to store the variance of each feature for each class.
self._priors: Array to store the prior probabilities for each class.
fit(X, y): Method to train the model by calculating the mean, variance, and priors for each class.
predict(X): Method to predict the class labels for the input data.
_predict(x): Helper method to predict the class label for a single sample.
_pdf(class_idx, x): Helper method to calculate the probability density function for a given class and sample based on Gaussian distribution.
'''