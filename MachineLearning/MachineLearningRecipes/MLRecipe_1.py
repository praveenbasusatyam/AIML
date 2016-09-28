
#
# Praveen Basu Satyam
# Hello World - Machine Learning Recipes #1
# https://www.youtube.com/watch?v=cKxRvEZd3Mw
# http://scikit-learn.org/
# http://scikit-learn.org/stable/modules/tree.html
# Supervised Learning - Classification
#

# features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
# labels = ["apple", "apple", "orange", "orange"]
#
# smooth = 1,
# bumpy = 0
#
# apple = 0
# orange = 1
#


#
from sklearn import tree

#
def decision_tree():
    print ""
    print "decision_tree()"
    print ""
    #
    features = [[140, 1], [130, 1], [150, 0], [170, 0]]
    labels = [0, 0, 1, 1]

    #
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, labels)

    #
    print clf.predict([150, 0])



#
decision_tree()