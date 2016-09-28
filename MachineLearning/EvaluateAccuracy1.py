
#
# http://scikit-learn.org/stable/modules/cross_validation.html
# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
#

#
# In this and the following exercises, you'll be adding train test splits to the data
# to see how it changes the performance of each classifier
#
# The code provided will load the Titanic dataset like you did in project 0, then train
# a decision tree (the method you used in your project) and a Bayesian classifier (as
# discussed in the introduction videos). You don't need to worry about how these work for
# now.
#
# What you do need to do is import a train/test split, train the classifiers on the
# training data, and store the resulting accuracy scores in the dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

# Limit to numeric data
X = X._get_numeric_data()
print X

# Separate the labels
y = X['Survived']
print y

# Remove labels from the inputs, and age due to missing data
del X['Age'], X['Survived']
print X

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

#
from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

#
print features_train.shape, labels_train.shape
print features_test.shape, labels_test.shape



def naive_Bayes_classifier(features_train, labels_train, features_test, labels_test):
    # The naive Bayes classifier
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    clf_accuracy_score = accuracy_score(labels_test, predictions)
    print "GaussianNB has accuracy: ", clf_accuracy_score
    return clf_accuracy_score


def decision_tree_classifier(features_train, labels_train, features_test, labels_test):
    # The decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    clf_accuracy_score = accuracy_score(labels_test, predictions)
    print "Decision Tree has accuracy: ", clf_accuracy_score
    return clf_accuracy_score


naive_Bayes_classifier_accuracy_score = naive_Bayes_classifier(features_train, labels_train, features_test, labels_test)
decision_tree_classifier_accuracy_score = decision_tree_classifier(features_train, labels_train, features_test, labels_test)

answer = {
 "Naive Bayes Score": naive_Bayes_classifier_accuracy_score,
 "Decision Tree Score": decision_tree_classifier_accuracy_score
}

print answer