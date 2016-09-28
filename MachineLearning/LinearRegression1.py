import numpy
import matplotlib.pyplot as plt


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
ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()



from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

### get Katie's net worth (she's 27)
### sklearn predictions are returned in an array, so you'll want to index into
### the output to get what you want, e.g. net_worth = predict([[27]])[0][0] (not
### exact syntax, the point is the [0] at the end). In addition, make sure the
### argument to your prediction function is in the expected format - if you get
### a warning about needing a 2d array for your data, a list of lists will be
### interpreted by sklearn as such (e.g. [[27]]).

### fill in the line of code to get the right value
km_net_worth = reg.predict([[27]])[0][0]

### get the slope
### again, you'll get a 2-D array, so stick the [0][0] at the end
### fill in the line of code to get the right value
slope = reg.coef_[0][0]

### get the intercept
### here you get a 1-D array, so stick [0] on the end to access
### the info we want
### fill in the line of code to get the right value
intercept = reg.intercept_[0]


### get the score on test data
### fill in the line of code to get the right value
test_score = reg.score(ages_test, net_worths_test)


### get the score on the training data
### fill in the line of code to get the right value
training_score = reg.score(ages_train, net_worths_train)



def submitFit():
    # all of the values in the returned dictionary are expected to be
    # numbers for the purpose of the grader.
    return {"networth":km_net_worth,
            "slope":slope,
            "intercept":intercept,
            "stats on test":test_score,
            "stats on training": training_score}


print submitFit()


###########

def regression_plot():

    #
    def ageNetWorthData_plot():
        random.seed(42)
        numpy.random.seed(42)

        ages = []
        for ii in range(100):
            ages.append( random.randint(20,65) )
        net_worths = [ii * 6.25 + numpy.random.normal(scale=40.) for ii in ages]
        ### need massage list into a 2d numpy array to get it to work in LinearRegression
        ages = numpy.reshape( numpy.array(ages), (len(ages), 1))
        net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

        return ages, net_worths



    ages, net_worths = ageNetWorthData_plot()
    reg_p = LinearRegression()
    reg_p.fit(ages, net_worths)


    #
    import matplotlib.pyplot as plt
    plt.scatter(ages, net_worths)
    plt.plot(ages, reg_p.predict(ages), color="blue", linewidth=3)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()

#
regression_plot()