import numpy as np
import pandas as pd

# Load the dataset
from sklearn.datasets import load_linnerud

linnerud_data = load_linnerud()
X = linnerud_data.data
y = linnerud_data.target

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)


reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
meas_dp = mae(y_test, predictions)
print "Decision Tree mean absolute error: {:.2f}".format(meas_dp)

reg = LinearRegression()
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
meas_lr = mae(y_test, predictions)
print "Linear regression mean absolute error: {:.2f}".format(meas_lr)

results = {
 "Linear Regression": meas_lr,
 "Decision Tree": meas_dp
}

print results