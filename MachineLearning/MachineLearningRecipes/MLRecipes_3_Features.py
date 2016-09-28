
#
# Praveen Basu Satyam
# What Makes a Good Feature? - Machine Learning Recipes #3
# https://www.youtube.com/watch?v=N9fDIAflCMY
# http://matplotlib.org/examples/statistics/histogram_demo_multihist.html
# Supervised Learning
#

import numpy as np
import matplotlib.pyplot as plt


def features_hist_plot():
    greyhounds = 500
    labs = 500

    grey_height = 28 + 4 * np.random.randn(greyhounds)
    lab_height = 24 + 4 * np.random.rand(labs)

    print ""
    print grey_height
    print lab_height

    plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
    plt.show()



#
features_hist_plot()