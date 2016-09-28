# As usual, use a train/test split to get a reliable F1 score from two classifiers, and
# save it the scores in the provided dictionaries.

#
# F1 = 2 * (precision * recall) / (precision + recall)
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
#

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)


def DecisionTreeClassifier_f1():
    #
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    #
    f1score = f1_score(y_test, predictions)
    #f1score = f1_score(predictions, y_test)
    print "Decision Tree F1 score: {:.2f}".format(f1score)
    return f1score


def GaussianNB_f1():
    #
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    #
    f1score = f1_score(y_test, predictions)
    #f1score = f1_score(predictions, y_test)
    print "GaussianNB F1 score: {:.2f}".format(f1score)
    return f1score


#
f1score_nb = GaussianNB_f1()
f1score_dt = DecisionTreeClassifier_f1()


#
F1_scores = {
 "Naive Bayes": f1score_nb,
 "Decision Tree": f1score_dt
}

print F1_scores