__author__ = 'Praveenb'


# Hello World
def hello_world():
    myvar = 'Hello World'
    print(myvar)


# file read
def file_open():
    with open('1.html') as f:
        print(f.read())


# data types
def data_types():
    x = 12
    y = 12.5
    n = "Hello"
    n2 = 'c'
    n3 = str('World')
    print(type(x))
    print(type(y))
    print(type(n))
    print(type(n2))
    print(type(n3))


# List
def list_op():
    l1 = [1, 2, 3]
    l2 = list()
    l3 = list([10, 20, 30, 40, 50, 60, 70, 80])
    l4 = list('python')
    print(type(l1))
    print(l2)
    print(l3)
    print(l4)

    # index
    print(l4[2])

    # in, not in
    print(10 in l3)
    print(33 not in l3)

    # func
    print(len(l3))
    print(max(l3))
    print(min(l3))
    print(sum(l3))

    # slice
    print(l3[0:4])
    print(l3[0:])
    print(l3[:5])

    # ope
    print(l3 + l1)
    print(l1 * 3)

    # traverse
    for i in l3:
        print(i)
        print()


# Dictionaries
def dict_op():
    dict1 = dict(k1='v1', k2='v2', k3='v3')
    for key in dict1:
        print(key, ':', dict1[key])
    print(dict1)
    print(type(dict1))


# tuple
def tuple_op():
    t1 = (1, 2, 3)
    t2 = tuple((10, 20, 30))
    print(t1)
    print(type(t2))
    for i in t2:
        print(i)
    print()


# if else
def if_else():
    i = 10
    if i % 2 == 0:
        print('Even')
    else:
        print('Odd')
