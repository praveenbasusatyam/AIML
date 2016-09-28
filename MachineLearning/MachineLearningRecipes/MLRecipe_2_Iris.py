

#
# Praveen Basu Satyam
# Visualizing a Decision Tree - Machine Learning Recipes #2
# https://www.youtube.com/watch?v=tNa99PG8hR8
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# http://scikit-learn.org/stable/datasets/
# http://scikit-learn.org/stable/modules/tree.html
# http://graphviz.org/?utm_campaign=chrome_series_decisiontree_041416&utm_source=gdev&utm_medium=yt-annt
# Supervised Learning
#

#
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree

#
def iris_pandas():
    print ""
    print "iris_pandas()"
    print ""
    #
    iris = load_iris()
    iris_pd = pd.DataFrame(pd.Series(iris))
    print iris_pd
    print pd.DataFrame(iris.data)



#
def iris_print_job():
    print ""
    print "iris_print_job()"
    print ""
    #
    iris = load_iris()
    print iris.feature_names
    print iris.target_names
    print iris.data[0]
    print iris.target[0]
    #
    for i in range(len(iris.target)):
        print "Example %d: Label %s, features %s" % (i, iris.target[i], iris.data[i])


#
def iris_decision_tree():
    print ""
    print "iris_decision_tree()"
    print ""

    iris = load_iris()
    test_idx = [0, 50, 100]

    # training data
    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data, test_idx, axis=0)
    print ""
    print train_data
    print train_target

    #test data
    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]
    print ""
    print test_data
    print test_target

    # tree training
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)

    # test
    print test_target
    predict = clf.predict(test_data)
    print predict

    # # TODO: Not working - graph.write_pdf("iris.pdf") - AttributeError: 'list' object has no attribute 'write_pdf'
    # from sklearn.externals.six import StringIO
    # import pydot
    #
    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,
    #                          feature_names=iris.feature_names,
    #                          class_names=iris.target_names,
    #                          filled=True, rounded=True,
    #                          special_characters=True)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("iris.pdf")


    #TODO: Image(graph.create_png()) AttributeError: 'list' object has no attribute 'create_png'
    from IPython.display import Image
    from sklearn.externals.six import StringIO
    import pydot

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                              feature_names=iris.feature_names,
                              class_names=iris.target_names,
                              filled=True, rounded=True,
                              special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    #Image(graph.create_png())


    #
    print ""
    print test_data[0], test_target[0]
    print iris.feature_names, iris.target_names

    return clf





# TODO: Not working - graph.write_pdf("iris.pdf") - AttributeError: 'list' object has no attribute 'write_pdf'
def decision_tree_viz(clf):
    # from sklearn.externals.six import StringIO
    # import pydot
    #
    # iris = load_iris()
    #
    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,
    #                          feature_names=iris.feature_names,
    #                          class_names=iris.target_names,
    #                          filled=True, rounded=True,
    #                          special_characters=True)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("iris.pdf")


    from sklearn.externals.six import StringIO
    import pydot
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("iris.pdf")





#
iris_pandas()

#
iris_print_job()

#
clf = iris_decision_tree()

#
#decision_tree_viz(clf)