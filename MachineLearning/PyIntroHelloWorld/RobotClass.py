__author__ = 'Praveenb'


# name : Public - These attributes can be freely used inside or outside of a class definition.
# _name : Protected - Protected attributes should not be used outside of the class definition, unless inside of a subclass definition.
# __name : Private - This kind of attribute is inaccessible and invisible. It's neither possible to read nor write to those attributes, except inside of the class definition itself.


class Robot:
    # class attribute
    a = "This is class attribute"
    counter = 0
    __counter = 0

    def __init__(self, name, build_year):
        print(name + " has been created")
        type(self).counter += 1
        type(self).__counter += 1
        # private attributes
        self.__name = name
        self.__build_year = build_year

    def __del__(self):
        type(self).counter -= 1
        type(self).__counter -= 1
        print(self.__name + " has been destroyed")

    # The output of __repr__ should be - if feasible -
    # a string which can be parsed by the python interpreter.
    # The result of this parsing is in an equal object.
    # This means that the following should be true for an object "o":
    # o == eval(repr(o))
    def __repr__(self):
        return "Robot(\"" + self.__name + "\"," + str(self.__build_year) + ")"

    def __str__(self):
        return "Hello from " + self.__name + ", Iam built in year " + str(self.__build_year)

    # static method
    @staticmethod
    def staticInstances():
        return Robot.__counter

    # class method
    @classmethod
    def classInstances(cls):
        return cls.__counter

    # public methods
    def sayHello(self):
        print("Hello from " + self.__name + ", Iam built in year " + str(self.__build_year))

    def setName(self, name):
        self.__name = name

    def setBuildYear(self, build_year):
        self.__build_year = build_year

    def getName(self):
        return self.__name

    def getBuildYear(self):
        return self.__build_year
