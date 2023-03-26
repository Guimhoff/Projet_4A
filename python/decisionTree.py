from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

class DecisionTree(BaseEstimator, ClassifierMixin):

    tree_ = []              # Shaped [variable, pivotValue, [variable, pivotValue, [variable, pivotValue, [class], [class]], [class]], ...]
    classNumber_ = None     # Stores number of class detected
    varNumber_ = None       # Stores number of variables detected
    logn_ = None            # Stores logarithm quotient used for entropy calculation
    maxDepth = None         # Maximum depth of the tree

    # Constructor method
    def __init__(self, maxDepth=999):
        # Parameters: maxDepth: int, maximum depth of the tree.
        self.maxDepth = maxDepth
        return

    ### Fitting

    # Training method
    def fit(self, X, y):
        # Parameters: X: array-like, shape (n_samples, n_features), the training input data, y: array-like, shape (n_samples,), the target values.
        # Returns: self: This decision tree object.

        self.countClass(y)
        self.countVar(X)
        
        # Adds the labels and features together into one array for easier processing
        trainValues = np.hstack((np.array(y).reshape(-1, 1), np.array(X)))

        self.tree_ = self.buildTree(trainValues)

        return self
    
    # Method to count number of unique classes in target values
    def countClass(self, y):
        # Parameters: y: list, target values.

        self.classNumber_ = max(y) + 1
        self.logn_ = np.log(self.classNumber_)
        return

    # Method to count and store number of variables
    def countVar(self, X):
        # Parameters: X: array-like, shape (n_samples, n_features), the provided dataset.
        
        self.varNumber_ = len(X[0])
        return

    # Method to build the decision tree recursively
    def buildTree(self, X, depth=0):
        # Parameters: part: array-like, shape (n_samples, n_features), partition of the provided dataset.
        # Results: tree: decision tree corresponding to the provided dataset partition.

        entropyParent = self.entropy(X)
        sizeParent = len(X)
        if entropyParent == 0:
            # Stopping criteria (leaf of the tree)
            return [int(X[0][0])]
        
        if depth >= self.maxDepth:
            # Stopping criteria (maximum depth reached)
            return [int(self.mostCommonClass(X))]

        bestVar = None
        bestVal = None
        bestEntropyGain = 0
        bestChild1 = []
        bestChild2 = []
        # Loop through all variables and examples to find the best split
        for var in range(self.varNumber_):
            for example in X:
                list1, list2 = self.split(var, example[1+var], X)
                entropyGain = entropyParent - len(list1)/sizeParent * self.entropy(list1) - len(list2)/sizeParent * self.entropy(list2)
                if entropyGain > bestEntropyGain:
                    bestVar = var
                    bestVal = example[1+var]
                    bestEntropyGain = entropyGain
                    bestChild1 = list1
                    bestChild2 = list2
        # Adding the best split the best split to the tree
        return [bestVar, bestVal, self.buildTree(bestChild1, depth=depth+1), self.buildTree(bestChild2, depth=depth+1)]

    # Method to calculate class proportions in a partition
    def proportions(self, partition):
        # Parameters: part: array-like, shape (n_samples, n_features), the given partition.
        # Results: prop: array-like, shape (self.classNumber_), the proportion of each class in the given partition.

        prop = np.array([0] * self.classNumber_)
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
    def mostCommonClass(self, X):
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
            output.append(self.browseTree(sample, self.tree_))

        return output
    
    # Recursively browse the decision tree to predict the class label of the sample.
    def browseTree(self, sample, tree):
        # Parameters: sample: array-like, shape (n_features,), the input data sample to predict; tree: list, the remaining part of the decision tree to browse.
        # Returns: ouptut: int, the predicted class label of the sample.

        if len(tree) == 1:
            # Stopping criteria (we are in a leaf)
            return tree[0]
        else:
            # Checking wich side of the tree we should follow
            if sample[tree[0]] <= tree[1]:
                return self.browseTree(sample, tree[2])
            else:
                return self.browseTree(sample, tree[3])
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
    testTree = DecisionTree().fit(X_train, y_train)
    print(testTree.tree_)
    prediction = testTree.predict(X_test)
    #print(prediction)
    #print(y_test)

    for n in range(len(prediction)):
        print(prediction[n] == y_test[n])
    
    print(testTree.score(X_test, y_test))

# Executed if not used as a dependency
if __name__ == '__main__':
    example()
