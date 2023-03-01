from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
import numpy as np
from collections import Counter

class DecisionTree(BaseEstimator, ClassifierMixin):

    tree_ = []           # de forme [variable, valeur, [variable, valeur, [variable, valeur, [classe], [classe]], [classe]], ...]
    classNumber_ = None
    varNumber_ = None
    logn_ = None

    def __init__(self):
        return

    # Fitting

    def fit(self, X, y):
        self.countClass(y)
        self.countVar(X)

        trainValues = np.hstack((np.array(y).reshape(-1, 1), np.array(X)))

        self.tree_ = self.buildTree(trainValues)

        return self
    
    def countClass(self, y):
        self.classNumber_ = len(Counter(y))
        self.logn_ = np.log(self.classNumber_)
        return

    def countVar(self, X):
        self.varNumber_ = len(X[0])
        return

    def buildTree(self, X):
        entropyParent = self.entropy(X)
        sizeParent = len(X)
        if entropyParent == 0:
            return [X[0][0]]
        else:
            bestVar = None
            bestVal = None
            bestEntropyGain = 0
            bestChild1 = []
            bestChild2 = []
            for var in range(self.varNumber_):
                for exemple in X:
                    liste1, liste2 = self.split(var, exemple[1+var], X)
                    entropyGain = entropyParent - len(liste1)/sizeParent * self.entropy(liste1) - len(liste2)/sizeParent * self.entropy(liste2)
                    if entropyGain > bestEntropyGain:
                        bestVar = var
                        bestVal = exemple[1+var]
                        bestEntropyGain = entropyGain
                        bestChild1 = liste1
                        bestChild2 = liste2
            
            return [bestVar, bestVal, self.buildTree(bestChild1), self.buildTree(bestChild2)]
        return

    def proportions(self, partition):
        prop = np.array([0] * self.classNumber_)
        for x in partition:
            prop[int(x[0])] += 1
        prop = prop / len(partition)
        return prop

    def entropy(self, part):
        prop = self.proportions(part)
        sortie = 0
        for x in prop:
            if x != 0:
                sortie += - x * np.log(x) / self.logn_
        return sortie

    def split(self, var, val, X):
        list1 = X[X[:, var+1] <= val]
        list2 = X[X[:, var+1] > val]
        return list1, list2
    
    # Predicting

    def predict(self, X):
        return []
    

iris = load_iris()
testTree = DecisionTree().fit(iris['data'], iris['target'])
print(testTree.tree_)