

#
# Praveen Basu Satyam
# Writing Our First Classifier - Machine Learning Recipes #5
# https://www.youtube.com/watch?v=AoeEHqVSNOw
#


import random


#
def euc(a,b):
    from scipy.spatial import distance
    return distance.euclidean(a,b)


#
class OwnKNN():
    def fit(self, X_train, Y_train):
        print "OwnKNN.fit()"
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        print "OwnKNN.predict()"
        predictions = []

        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions


    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if (dist < best_dist):
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]


#
def accuracy_test_own_KNeighborsClassifier():
    from sklearn import datasets
    iris = datasets.load_iris()

    X = iris.data
    Y = iris.target

    #
    from sklearn.cross_validation import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

    #
    my_classifier = OwnKNN()

    my_classifier.fit(X_train, Y_train)
    predictions = my_classifier.predict(X_test)

    #
    from sklearn.metrics import accuracy_score
    print accuracy_score(Y_test, predictions)
    print ""



#
accuracy_test_own_KNeighborsClassifier()