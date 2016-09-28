__author__ = 'Praveenb'

from functools import reduce


class MapReduceLambdaFilter:
    def __init__(self):
        self.__id = ''

    def __del__(self):
        print('')

    def lambda_map(self):
        lambda_fahrenheit = lambda x: ((float(9) / 5) * x + 32)
        lambda_celsius = lambda x: ((float(5) / 9) * (x - 32))
        temperatures = (36.5, 37, 35, 39)
        #
        fahrenheit_temp = list(map(lambda_fahrenheit, temperatures))
        print('MapReduceLambdaFilter.lambda_map().fahrenheit_temp : ')
        print(fahrenheit_temp)
        #
        celsius_temp = list(map(lambda_celsius, fahrenheit_temp))
        print('MapReduceLambdaFilter.lambda_map().celsius_temp : ')
        print(celsius_temp)

    def lambda_map_multiple(self):
        a = [1, 2, 3, 4, 5]
        b = [10, 20, 30, 40, 50]
        c = [100, 200, 300, 300, 400, 500]
        lambda_func = lambda x, y, z: x + y + z
        lambda_transform = list(map(lambda_func, a, b, c))
        print('MapReduceLambdaFilter.lambda_map_multiple() : ')
        print(lambda_transform)

    def lambda_filter(self):
        fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21]
        even_list = list(filter((lambda x: x % 2 == 0), fibonacci))
        odd_list = list(filter((lambda x: x % 2), fibonacci))
        print('MapReduceLambdaFilter.lambda_filter().even_list : ')
        print(even_list)
        print('MapReduceLambdaFilter.lambda_filter().odd_list : ')
        print(odd_list)

    def lambda_reduce(self):
        list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        reduce_add = reduce((lambda x, y: x + y), list)
        reduce_mul = reduce((lambda x, y: x * y), list)
        reduce_mul2 = reduce((lambda x, y: x * y), range(1, 1000))
        print('MapReduceLambdaFilter.lambda_reduce().reduce_add : ')
        print(reduce_add)
        print('MapReduceLambdaFilter.lambda_reduce().reduce_mul : ')
        print(reduce_mul)
        print('MapReduceLambdaFilter.lambda_reduce().reduce_mul2 : ')
        print(reduce_mul2)
