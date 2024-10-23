import numpy as np
from collections import Counter
from decision_tree import DecisionTree  # Assuming the DecisionTree class is in a file named decision_tree.py

# Function to create a bootstrap sample from the dataset
def bootstrap_sample(X, y):
    n_sample = X.shape[0]  # Number of samples in the dataset
    idxs = np.random.choice(n_sample, size=n_sample, replace=True)  # Randomly sample with replacement
    return X[idxs], y[idxs]  # Return the sampled dataset

# Function to find the most common label in an array
def most_common_label(y):
    counter = Counter(y)  # Count the occurrences of each label
    return counter.most_common(1)[0][0]  # Return the most common label

# Random Forest Classifier
class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees  # Number of trees in the forest
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split
        self.max_depth = max_depth  # Maximum depth of each tree
        self.n_feats = n_feats  # Number of features to consider for splits
        self.trees = []  # List to store the trees

    # Function to train the random forest
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)  # Create a bootstrap sample
            tree.fit(X_sample, y_sample)  # Train the tree on the bootstrap sample
            self.trees.append(tree)  # Add the trained tree to the list of trees

    # Function to make predictions
    def predict(self, X):
        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])  
        tree_preds = np.swapaxes(tree_preds, 0, 1)  # Transpose the array to have shape (n_samples, n_trees)

        # Aggregate predictions by taking the most common label among the trees
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)  # Return the aggregated predictions


'''
Summary of Comments:
Bootstrap Sampling: Explained the process of creating a bootstrap sample from the dataset.
Most Common Label: Described how the most common label is determined in an array.
RandomForest Class:
Initialization: Explained the parameters used for initializing the random forest.
fit Method: Described the process of training the random forest by creating bootstrap samples and training decision trees on them.
predict Method: Detailed the process of making predictions by aggregating the predictions from all the decision trees in the forest.
'''