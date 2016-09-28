
#
# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

#
# Recall: True Positive / (True Positive + False Negative)
# Precision: True Positive / (True Positive + False Positive).
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
#

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)


def DecisionTreeClassifier_pr():
    #
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    #
    #recalls = recall(y_test, predictions)
    #precisions = precision(y_test, predictions)
    recalls = recall(predictions, y_test)
    precisions = precision(predictions, y_test)
    print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(recalls, precisions)
    return recalls, precisions


def GaussianNB_pr():
    #
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    #
    #recalls = recall(y_test, predictions)
    #precisions = precision(y_test, predictions)
    recalls = recall(predictions, y_test)
    precisions = precision(predictions, y_test)
    print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(recalls, precisions)
    return recalls, precisions


#
nb_recall, nb_precision = GaussianNB_pr()
dt_recall, dt_precision = DecisionTreeClassifier_pr()


results = {
  "Naive Bayes Recall": nb_recall,
  "Naive Bayes Precision": nb_precision,
  "Decision Tree Recall": dt_recall,
  "Decision Tree Precision": dt_precision
}

print results