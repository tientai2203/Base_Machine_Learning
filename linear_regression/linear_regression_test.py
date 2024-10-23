import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1,noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

# print(X.shape)
# print(X[:, 0])
# print(X)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], y, color = "b", marker="o", s=20)
# plt.show()

from linear_regression import LinearRegression
# generate an instance of LinearRegression
regressor = LinearRegression(lr = 0.001, iters = 10000)

# fit with datasets and gradients
regressor.fit(X_train, y_train)

# predict
predicted = regressor.predict(X_test)

# cost function = MSE_function
def MSE(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

mse_value = MSE(y_test, predicted)
# print(loss)
print(mse_value)

# Assuming regressor is an instance of LinearRegression and already trained
y_pred_line = regressor.predict(X)

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))

# Plot the training data
m1 = plt.scatter(regressor.X_train, regressor.y_train, color=cmap(0.9), s=10, label='Training Data')

# Plot the test data
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label='Test Data')

# Plot the prediction line
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")

# Adding labels and title
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression Prediction')

# Display the plot
plt.show()
