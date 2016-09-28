__author__ = 'Praveenb'


#
from Main1 import *
import datetime
from RobotClass import Robot
from ClassGettersSetters import GS
from ClassInheritance import Employee
from MapReduceLambdaFilter import MapReduceLambdaFilter


class Main:
    #
    def __init__(self):
        print('Main.__init__(self)')

    #
    def __del__(self):
        print('Main.__del__()')

    #
    def __repr__(self):
        return 'Main()'

    #
    def __str__(self):
        return 'Main class'

    #
    @staticmethod
    def test1():
        hello_world()
        file_open()
        data_types()
        list_op()
        dict_op()
        tuple_op()
        if_else()

    #
    @classmethod
    def repr_test(cls):
        # repr(), __repr()__
        # The output of __repr__ should be - if feasible -
        # a string which can be parsed by the python interpreter.
        # The result of this parsing is in an equal object.
        # This means that the following should be true for an object "o":
        # o == eval(repr(o))

        list_rt = [1, 2, 3, 4]
        list_rts = repr(list_rt)
        list_rtse = eval(list_rts)
        print(list_rt)
        print(type(list_rt))
        print(list_rts)
        print(type(list_rts))
        print(list_rtse)
        print(type(list_rtse))
        print(list_rt == list_rtse)

    # repr()
    @classmethod
    def repr_test2(cls):
        today = datetime.datetime.today()
        print(str(today))
        print(eval(repr(today)))
        print(type(eval(repr(today))))

    # Class
    @classmethod
    def class_test(cls):
        robox = Robot('mantle', 1975)
        roboy = Robot('gradle', 1980)
        robox.sayHello()
        roboy.sayHello()
        #
        for rob in [robox, roboy]:
            rob.sayHello()

    #
    @classmethod
    def eval_test(cls):
        robox = Robot('mantle', 1975)
        roboy = Robot('gradle', 1980)
        # o == eval(repr(o))
        print(robox)
        print(type(robox))
        print(str(robox))
        print(type(str(robox)))
        print(repr(roboy))
        print(type(repr(roboy)))
        print(type(eval(repr(roboy))))

    #
    @classmethod
    def call_test(cls):
        # For a Class C, an instance x of C and a method m of C the following three method calls are equivalent:
        # type(x).m(x, ...)
        # C.m(x, ...)
        # x.m(...)
        robox = Robot('mantle', 1975)
        roboy = Robot('gradle', 1980)

        type(robox).sayHello(robox)
        Robot.sayHello(robox)
        robox.sayHello()

    # del
    @classmethod
    def delete_test(cls):
        roboz = Robot('Po', 1990)
        del roboz

    # class attribute
    @classmethod
    def class_attributes(cls):
        robox = Robot('mantle', 1975)
        roboy = Robot('gradle', 1980)

        print(Robot.a)
        print(robox.a)
        roboy.a = 'this is now instance attribute'
        print(roboy.a)

        print(Robot.__dict__)
        print(robox.__dict__)
        print(roboy.__dict__)
        print(robox.__class__.__dict__)
        print(roboy.__class__.__dict__)

        # static / class methods
        print(Robot.counter)
        print(type(roboy).counter)
        print(Robot.staticInstances())
        print(roboy.staticInstances())
        print(type(roboy).staticInstances())
        print(Robot.classInstances())
        print(robox.classInstances())

    # property setters / getters
    @classmethod
    def getters_setters(cls):
        instance_x = GS(10000)
        print(instance_x.attribute_x)
        instance_x.attribute_x = -5000
        print(instance_x.attribute_x)
        print(instance_x.condition)

    # inheritance
    @classmethod
    def inheritance_test(cls):
        instance_x = Employee('John', '007')
        print(instance_x.get_employee_details())
        print(instance_x.details)

    # lambda, map
    @classmethod
    def map_reduce_lambda_filter_(cls):
        instance_x = MapReduceLambdaFilter()
        instance_x.lambda_map()
        instance_x.lambda_map_multiple()
        instance_x.lambda_filter()
        instance_x.lambda_reduce()



#
Main.test1()
Main.repr_test()
Main.repr_test2()
Main.class_test()
Main.eval_test()
Main.call_test()
Main.delete_test()
Main.class_attributes()
Main.getters_setters()
Main.inheritance_test()
Main.map_reduce_lambda_filter_()
