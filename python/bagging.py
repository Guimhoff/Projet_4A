from sklearn.base import BaseEstimator, ClassifierMixin
from decisionTree import DecisionTree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris

class Bagging(BaseEstimator, ClassifierMixin):

    tree_count = 4                   # Number of decision tree used
    single_tree_data_ratio = .8        # Ratio of data used in each tree relatively to the entire training dataSet
    max_depth = 999                  # Maximum depth of the tree

    tree_list_ = []                  # List of DecisionTree generated

    # Constructor method
    def __init__(self, tree_count=4, single_tree_data_ratio=.8, max_depth=999):
        # Parameters: tree_count: int, defines the number of DecisionTree generated by the estimator; dataRatio: float, between 0.0 and 1.0, original dataSet ratio used to train each individual DecisionTree.
        self.tree_count = tree_count
        self.single_tree_data_ratio = single_tree_data_ratio
        self.max_depth = max_depth
        return

    ### Fitting

    # Training method
    def fit(self, X, y):
        # Parameters: X: array-like, shape (n_samples, n_features), the training input data; y: array-like, shape (n_samples,), the target values.
        # Returns: self: This bagging object.

        for _ in range(self.tree_count):
            n_X, _, n_y, _ = train_test_split(X, y, test_size=1-self.single_tree_data_ratio)
            self.tree_list_.append(DecisionTree(self.max_depth).fit(n_X, n_y))

        return self
    
    ### Predicting

    # Prediction method
    def predict(self, X):
        # Parameters: X: array-like, shape (n_samples, n_features), the input data.
        # Returns: ouptut: list, the predicted class for the provided data.

        bag = []
        for tree in self.tree_list_:
            bag.append(tree.predict(X))
        
        output = []
        for n in range(len(X)):
            election = {}
            for k in range(self.tree_count):
                vote = bag[k][n]
                if vote in election.keys():
                    election[vote] += 1
                else:
                    election[vote] = 1
            output.append(max(election, key=election.get))

        return output
    
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
    test_bagging = Bagging(50, .6).fit(X_train, y_train)
    prediction = test_bagging.predict(X_test)
    #print(prediction)
    #print(y_test)

    for n in range(len(prediction)):
        print(prediction[n] == y_test[n])
    print(test_bagging.score(X_test, y_test))

# Executed if not used as a dependency
if __name__ == '__main__':
    example()
