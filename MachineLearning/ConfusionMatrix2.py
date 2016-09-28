
#
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#

# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
print X

y = X['Survived']
print y

del X['Age'], X['Survived']
print

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
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
    clf_confusion_matrix = confusion_matrix(labels_test, predictions)
    print "GaussianNB confusion matrix:\n", clf_confusion_matrix
    return clf_confusion_matrix


def decision_tree_classifier(features_train, labels_train, features_test, labels_test):
    # The decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    clf_confusion_matrix = confusion_matrix(labels_test, predictions)
    print "Confusion matrix for this Decision Tree:\n ", clf_confusion_matrix
    return clf_confusion_matrix


naive_Bayes_classifier_confusion_matrix = naive_Bayes_classifier(features_train, labels_train, features_test, labels_test)
decision_tree_classifier_confusion_matrix = decision_tree_classifier(features_train, labels_train, features_test, labels_test)

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": naive_Bayes_classifier_confusion_matrix,
 "Decision Tree": decision_tree_classifier_confusion_matrix
}

print confusions