

print(__doc__)


from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.cross_validation import KFold
import numpy as np

import numpy
import random

def ageNetWorthData():

    random.seed(42)
    numpy.random.seed(42)

    ages = []
    for ii in range(100):
        ages.append( random.randint(20,65) )
    net_worths = [ii * 6.25 + numpy.random.normal(scale=40.) for ii in ages]
    ### need massage list into a 2d numpy array to get it to work in LinearRegression
    ages = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    from sklearn.cross_validation import train_test_split
    ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths)

    return ages_train, ages_test, net_worths_train, net_worths_test


#from ages_net_worths import ageNetWorthData
X_train, X_test, y_train, y_test = ageNetWorthData()


# size = 1000
# cv = KFold(size, shuffle=True)
# score = make_scorer(explained_variance_score)
#
# X = np.reshape(np.random.normal(scale=2, size=size), (-1, 1))
# y = np.array([[1 - 2 * x[0] + x[0] ** 2] for x in X])
#
#
# from sklearn import cross_validation
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)


def linear_regression():
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg



def plot_cruve(predictions):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.scatter(X_train, y_train, color="b", label="train data")
    plt.scatter(X_test, y_test, color="r", label="test data")
    plt.plot(y_test, predictions, color="black")
    plt.legend(loc=2)
    plt.xlabel("ages")
    plt.ylabel("net worths")

    # Sizes the window for readability and displays the plot.
    #plt.ylim(-.1, 1.1)
    plt.show()

    #plt.savefig("test.png")
    #output_image("test.png", "png", open("test.png", "rb").read())



reg = linear_regression()
predictions = reg.predict(X_test)

plot_cruve(predictions)


#predict
print "test predciton : ", reg.predict([[27]])
print "slope : ", reg.coef_
print "intercept : ", reg.intercept_

print "r-squared score (test) : ", reg.score(X_test, y_test)
print "r-squared score (train) : ", reg.score(X_train, y_train)

