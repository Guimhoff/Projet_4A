from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from random import randint

class DecisionTree(BaseEstimator, ClassifierMixin):

    tree_ = []              # Shaped [variable, pivotValue, [variable, pivotValue, [variable, pivotValue, [class], [class]], [class]], ...]
    class_number_ = None     # Stores number of class detected
    var_number_ = None       # Stores number of variables detected
    logn_ = None            # Stores logarithm quotient used for entropy calculation
    max_depth = None         # Maximum depth of the tree

    # Constructor method
    def __init__(self, max_depth=999):
        # Parameters: max_depth: int, maximum depth of the tree.
        self.max_depth = max_depth
        return

    ### Fitting

    # Training method
    def fit(self, X, y):
        # Parameters: X: array-like, shape (n_samples, n_features), the training input data, y: array-like, shape (n_samples,), the target values.
        # Returns: self: This decision tree object.

        self.count_class(y)
        self.count_var(X)
        
        # Adds the labels and features together into one array for easier processing
        train_values = np.hstack((np.array(y).reshape(-1, 1), np.array(X)))

        self.tree_ = self.build_tree(train_values)

        return self
    
    # Method to count number of unique classes in target values
    def count_class(self, y):
        # Parameters: y: list, target values.

        self.class_number_ = max(y) + 1
        self.logn_ = np.log(self.class_number_)
        return

    # Method to count and store number of variables
    def count_var(self, X):
        # Parameters: X: array-like, shape (n_samples, n_features), the provided dataset.
        
        self.var_number_ = len(X[0])
        return

    # Method to build the decision tree recursively
    def build_tree(self, X, depth=0):
        # Parameters: part: array-like, shape (n_samples, n_features), partition of the provided dataset.
        # Results: tree: decision tree corresponding to the provided dataset partition.

        entropy_parent = self.entropy(X)
        size_parent = len(X)
        if entropy_parent == 0:
            # Stopping criteria (leaf of the tree)
            return [int(X[0][0])]
        
        if depth >= self.max_depth:
            # Stopping criteria (maximum depth reached)
            return [int(self.most_common_class(X))]

        best_var = None
        best_val = None
        best_entropy_gain = 0
        best_child_1 = []
        best_child_2 = []
        # Loop through all variables and examples to find the best split
        for var in range(self.var_number_):
            for example in X:
                list1, list2 = self.split(var, example[1+var], X)
                entropy_gain = entropy_parent - len(list1)/size_parent * self.entropy(list1) - len(list2)/size_parent * self.entropy(list2)
                if entropy_gain > best_entropy_gain:
                    best_var = var
                    best_val = example[1+var]
                    best_entropy_gain = entropy_gain
                    best_child_1 = list1
                    best_child_2 = list2
        # Adding the best split the best split to the tree
        return [best_var, best_val, self.build_tree(best_child_1, depth=depth+1), self.build_tree(best_child_2, depth=depth+1)]

    # Method to calculate class proportions in a partition
    def proportions(self, partition):
        # Parameters: part: array-like, shape (n_samples, n_features), the given partition.
        # Results: prop: array-like, shape (self.class_number_), the proportion of each class in the given partition.

        prop = np.array([0] * self.class_number_)
        for x in partition:
            prop[int(x[0])] += 1
        prop = prop / len(partition)
        return prop

    # Method to calculate entropy of a partition
    def entropy(self, part):
        # Parameters: part: array-like, shape (n_samples, n_features), the given partition.
        # Results: output: float, the calculated entropy.

        prop = self.proportions(part)
        output = 0
        for x in prop:
            if x != 0:
                output += - x * np.log(x) / self.logn_
        return output

    # Method to split a partition into two based on a variable and its value
    def split(self, var, val, X):
        # Parameters: var: int, the variable used for partition index; val: float, the separation value used to split the samples; X: array-like, shape (n_samples, n_features), the initial partition.
        # Returns: list1, list2: array-like, shape (n_samples, n_features), the 2 resulting partitions.

        list1 = X[X[:, var+1] <= val]
        list2 = X[X[:, var+1] > val]
        return list1, list2
    
    # Method to find the most common class in a partition
    def most_common_class(self, X):
        # Parameters: X: array-like, shape (n_samples, n_features), the initial partition.
        # Returns: output: int, the most common class in the partition.

        output = Counter(X[:, 0]).most_common(1)[0][0]
        return output

    ### Predicting

    # Prediction method
    def predict(self, X):
        # Parameters: X: array-like, shape (n_samples, n_features), the input data.
        # Returns: ouptut: list, the predicted class labels for the provided data.

        output = []

        for sample in X:
            output.append(self.browse_tree(sample, self.tree_))

        return output
    
    # Recursively browse the decision tree to predict the class label of the sample.
    def browse_tree(self, sample, tree):
        # Parameters: sample: array-like, shape (n_features,), the input data sample to predict; tree: list, the remaining part of the decision tree to browse.
        # Returns: ouptut: int, the predicted class label of the sample.

        if len(tree) == 1:
            # Stopping criteria (we are in a leaf)
            return tree[0]
        else:
            # Due to errors encountered, we check if the sample has the right number of features
            if len(sample) < tree[0]: 
                return self.browse_tree(sample, tree[randint(2,3)])

            # Checking wich side of the tree we should follow
            if sample[tree[0]] <= tree[1]:
                return self.browse_tree(sample, tree[2])
            else:
                return self.browse_tree(sample, tree[3])
        return

    # Score method
    def score(self, X, y):
        # Parameters: X: array-like, shape (n_samples, n_features), the input data; y: array-like, shape (n_samples,), the expected corresponding class.
        # Returns: score: Ratio of predictions matching the expected class.
        
        preds = self.predict(X)
        return np.mean(preds == y)
    

# Demonstration

# Demonstration of the DecisionTree classifier on the iris dataset.
def example():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'])
    test_tree = DecisionTree().fit(X_train, y_train)
    print(test_tree.tree_)
    prediction = test_tree.predict(X_test)
    #print(prediction)
    #print(y_test)

    for n in range(len(prediction)):
        print(prediction[n] == y_test[n])
    
    print(test_tree.score(X_test, y_test))

# Executed if not used as a dependency
if __name__ == '__main__':
    example()
