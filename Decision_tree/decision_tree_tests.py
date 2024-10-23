import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from decision_tree import DecisionTree

# load data from sklearn
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# split train/test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_pred==y_true)/len(y_true)
    return accuracy

regressor = DecisionTree(max_depth = 10)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print("Decision Tree classification accuracy: ", accuracy(y_test, predictions))
