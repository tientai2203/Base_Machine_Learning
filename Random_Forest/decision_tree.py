import numpy as np
from collections import Counter

# Function to calculate entropy of a label distribution
def entropy(y):
    hist = np.bincount(y)  # Count occurrences of each class label
    ps = hist / len(y)  # Calculate probabilities
    return -np.sum([p * np.log2(p) for p in ps if p > 0])  # Compute entropy

# Class to represent a node in the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # Feature index used for splitting
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value for leaf node (class label)

    def is_leaf_node(self):
        return self.value is not None  # Check if the node is a leaf node

# Decision Tree Classifier
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split
        self.max_depth = max_depth  # Maximum depth of the tree
        self.n_feats = n_feats  # Number of features to consider for splits
        self.root = None  # Root node of the tree

    # Function to train the decision tree
    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])  # Determine the number of features to use
        self.root = self._grow_tree(X, y)  # Grow the tree from the root

    # Recursive function to grow the tree
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape  # Number of samples and features
        n_labels = len(np.unique(y))  # Number of unique class labels

        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)  # Get the most common label
            return Node(value=leaf_value)  # Return a leaf node with the most common label

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)  # Randomly select features

        # Find the best split
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)  # Split the data
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)  # Grow the left subtree
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)  # Grow the right subtree

        return Node(best_feat, best_thresh, left, right)  # Return the current node

    # Function to find the best criteria (feature and threshold) for splitting
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1  # Initialize the best information gain
        split_idx, split_thresh = None, None  # Initialize the best feature index and threshold

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]  # Get the column of the current feature
            thresholds = np.unique(X_column)  # Get unique values in the feature column
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)  # Calculate information gain

                if gain > best_gain:
                    best_gain = gain  # Update the best gain
                    split_idx = feat_idx  # Update the best feature index
                    split_thresh = threshold  # Update the best threshold

        return split_idx, split_thresh  # Return the best feature index and threshold

    # Function to calculate information gain
    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = entropy(y)  # Calculate the entropy of the parent node
        left_idxs, right_idxs = self._split(X_column, split_thresh)  # Split the data

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0  # Return 0 if the split doesn't separate the data

        # Calculate the weighted average of the child entropies
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_l + (n_right / n) * e_r

        # Information gain is the reduction in entropy
        ig = parent_entropy - child_entropy
        return ig

    # Function to split the data based on a threshold
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()  # Indices of the left split
        right_idxs = np.argwhere(X_column > split_thresh).flatten()  # Indices of the right split
        return left_idxs, right_idxs

    # Function to make predictions
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])  # Traverse the tree for each sample

    # Recursive function to traverse the tree and make a prediction
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value  # Return the value if it's a leaf node

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)  # Traverse left if the feature value is less than or equal to the threshold
        return self._traverse_tree(x, node.right)  # Traverse right otherwise

    # Function to find the most common label in the data
    def _most_common_label(self, y):
        counter = Counter(y)  # Count the occurrences of each label
        return counter.most_common(1)[0][0]  # Return the most common label
    

'''
Summary of Comments:
Entropy Calculation: Explained the steps for calculating the entropy of a label distribution.
Node Class: Described the purpose of the class and the meaning of each attribute.
DecisionTree Class:
Initialization: Explained the parameters used for initializing the decision tree.
fit Method: Described the process of training the decision tree.
_grow_tree Method: Explained the recursive process of growing the tree.
_best_criteria Method: Described how the best feature and threshold for splitting are determined.
_information_gain Method: Explained the calculation of information gain.
_split Method: Detailed the process of splitting the data based on a threshold.
predict Method: Described how predictions are made using the trained tree.
_traverse_tree Method: Explained the recursive tree traversal for making predictions.
_most_common_label Method: Described how the most common label in the data is found.
'''
