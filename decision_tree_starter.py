"""
To prepare the starter code, copy this file over to decision_tree_starter.py
and go through and handle all the inline TODOs.
"""
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
from pydot import graph_from_dot_data
import io
import matplotlib.pyplot as plt

import random
SEED = 246810
random.seed(SEED)
np.random.seed(SEED)

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        self.labels = None

    @staticmethod
    def entropy(y):
        # TODO
        prob = np.where(y > 0.5)[0].size/len(y)
        return -prob*np.log2(prob+eps)-(1-prob)*np.log2(1-prob+eps)

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO
        # X must be a single feature vector
        idx_left = np.where(X < thresh)[0]
        idx_right = np.where(X >= thresh)[0]
        
        # redundant split case
        if idx_left.size==0 or idx_right.size==0:
            return 0
        
        H = DecisionTree.entropy
        y_left = y[idx_left]
        y_right = y[idx_right]
        H_after = (len(y_left)*H(y_left)+len(y_right)*H(y_right))/(len(y_left)+len(y_right))
        
        return H(y)-H_after
        

    @staticmethod
    def gini_impurity(y):
        # OPTIONAL
        prob = np.where(y < 0.5)[0].size/len(y)
        return 1.0-prob**2-(1-prob)**2

    @staticmethod
    def gini_purification(X, y, thresh):
        # OPTIONAL
        # X must be a single feature vector
        idx_left = np.where(X < thresh)[0]
        idx_right = np.where(X >= thresh)[0]
        
        # redundant split case
        if idx_left.size==0 or idx_right.size==0:
            return 0
        
        H = DecisionTree.gini_impurity
        y_left = y[idx_left]
        y_right = y[idx_right]
        H_after = (len(y_left)/len(y))*H(y_left)+(len(y_right)/len(y))*H(y_right)
        
        return H(y)-H_after

    def split(self, X, y, feature_idx, thresh):
        """
        Split the dataset into two subsets, given a feature and a threshold.
        Return X_0, y_0, X_1, y_1
        where (X_0, y_0) are the subset of examples whose feature_idx-th feature
        is less than thresh, and (X_1, y_1) are the other examples.
        """
        # TODO
        idx_left = np.where(X[:, feature_idx]<thresh)[0]
        idx_right = np.where(X[:, feature_idx]>=thresh)[0]
        X_0, y_0 = X[idx_left, :], y[idx_left]
        X_1, y_1 = X[idx_right, :], y[idx_right]
        return X_0, y_0, X_1, y_1

    def fit(self, X, y, gini=False):
        # TODO
        np.random.seed(SEED)
        if self.max_depth > 0:
            best_gain = 0
            best_split = None
            threshs = np.array([np.linspace(np.min(X[:, i]+eps), np.max(X[:, i]-eps), 10) for i in range(X.shape[1])])
            # threshs = np.array([np.unique(X[:, i]) for i in range(X.shape[1])])  # could be too many
            if gini:
                for f_idx in range(X.shape[1]):
                    if X[:, f_idx].size == 0 or np.all(X[:, f_idx]==X[:, f_idx][0]):
                        continue
                    
                    info_gains = []
                    for thresh in threshs[f_idx, :]:
                        info_gains.append(DecisionTree.gini_purification(X[:, f_idx], y, thresh))
                    
                    best_local_idx = np.argmax(info_gains)
                    best_local_gain = info_gains[best_local_idx]
                    
                    if best_local_gain>best_gain:
                        best_gain = best_local_gain
                        best_split = (f_idx, threshs[f_idx][best_local_idx])
            else:
                for f_idx in range(X.shape[1]):
                    if X[:, f_idx].size == 0 or np.all(X[:, f_idx]==X[:, f_idx][0]):
                        continue
                    
                    info_gains = []
                    for thresh in threshs[f_idx, :]:
                        info_gains.append(DecisionTree.information_gain(X[:, f_idx], y, thresh))
                        
                    best_local_idx = np.argmax(info_gains)
                    best_local_gain = info_gains[best_local_idx]
                    
                    if best_local_gain>best_gain:
                        best_gain = best_local_gain
                        best_split = (f_idx, threshs[f_idx][best_local_idx])
            
            # stopping criterion
            if best_split is None or best_gain < eps:
                self.pred = stats.mode(y, keepdims=True).mode[0]
                self.max_depth = 0
                self.labels = y
                return self
            
            self.split_idx, self.thresh = best_split
            X_0, y_0, X_1, y_1 = self.split(X, y, self.split_idx, self.thresh)
            self.left = DecisionTree(max_depth=self.max_depth-1, feature_labels=self.features)
            self.left.fit(X_0, y_0)
            self.right = DecisionTree(max_depth=self.max_depth-1, feature_labels=self.features)
            self.right.fit(X_1, y_1)
    
        else:
            self.max_depth = 0
            self.split_idx = 0
            self.data= X
            self.labels = y
            if len(y)>0:
                self.pred = stats.mode(y, keepdims=True).mode[0]
            else:
                self.pred = 0
            
        return self

    def predict(self, X):
        # TODO
        # for the parent node
        if self.max_depth > 0:
            idx_left = np.where(X[:, self.split_idx]<self.thresh)[0]
            idx_right = np.where(X[:, self.split_idx]>=self.thresh)[0]
            X_0 = X[idx_left, :]
            X_1 = X[idx_right, :]
            preds = np.zeros(X.shape[0])
            preds[idx_left] = self.left.predict(X_0)
            preds[idx_right] = self.right.predict(X_1)
            return preds
        
        # for leaf nodes
        else:
            self.split_idx = 0
            return np.array([self.pred]*X.shape[0])

    def _to_graphviz(self, node_id):
        if self.max_depth == 0:
            return f'{node_id} [label="Prediction: {self.pred}\nSamples: {self.labels.size}"];\n'
        else:
            if self.split_idx is None:
                self.split_idx = 0
            graph = f'{node_id} [label="{self.features[self.split_idx]} < {self.thresh:.2f}"];\n'
            left_id = node_id * 2 + 1
            right_id = node_id * 2 + 2
            if self.left is not None:
                graph += f'{node_id} -> {left_id};\n'
                graph += self.left._to_graphviz(left_id)
            if self.right is not None:
                graph += f'{node_id} -> {right_id};\n'
                graph += self.right._to_graphviz(right_id)
            return graph

    def to_graphviz(self):
        graph = "digraph Tree {\nnode [shape=box];\n"
        graph += self._to_graphviz(0)
        graph += "}\n"
        return graph
        
    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            if self.split_idx is None:
                self.split_idx = 0
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO
        np.random.seed(SEED)
        for i in range(self.n):
            new_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            new_X, new_y = X[new_indices, :], y[new_indices]
            self.decision_trees[i].fit(new_X, new_y)
        
        return self

    def predict(self, X):
        # TODO
        pred = [self.decision_trees[i].predict(X) for i in range(self.n)]
        pred = np.round(np.mean(pred, axis=0))  # pred should include elements with values only 0 or 1
        return pred


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)


class BoostedRandomForest(RandomForest):
    # OPTIONAL
    def fit(self, X, y):
        # OPTIONAL
        pass
    
    def predict(self, X):
        # OPTIONAL
        pass


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # Temporarily assign -1 to missing data
    data[data == ''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == '-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        # TODO
        for i in range(data.shape[1]):
            mode = stats.mode(data[:, i]).mode
            data[data[:, i]==-1, i] = mode

    return data, onehot_features


def evaluate(clf):
    print("Cross validation", cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)


def generate_submission(testing_data, predictions, dataset="titanic"):
    assert dataset in ["titanic", "spam"], f"dataset should be either 'titanic' or 'spam'"
    # This code below will generate the predictions.csv file.
    if isinstance(predictions, np.ndarray):
        predictions = predictions.astype(int)
    else:
        predictions = np.array(predictions, dtype=int)
    assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
    df = pd.DataFrame({'Category': predictions})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(f'predictions_{dataset}.csv', index_label='Id')


if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=str)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=str)
        y = data[1:, 0].astype(str)  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != '')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    # Decision Tree
    print("\n\nDecision Tree")
    dt = DecisionTree(max_depth=3, feature_labels=features)
    dt.fit(X, y)

    # Visualize Decision Tree
    print("\n\nTree Structure")
    # Print using repr
    print(dt.__repr__())
    # Save tree to pdf
    graph_from_dot_data(dt.to_graphviz())[0].write_pdf("%s-basic-tree.pdf" % dataset)