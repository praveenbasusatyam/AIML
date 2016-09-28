__author__ = 'Praveenb'


#
class Person:
    def __init__(self, name):
        print('Person.__init__()')
        self.__name = ''
        self.name = name

    def __del__(self):
        print('Person.__del__()')

    @property
    def name(self):
        return self.__name

    @property
    def details(self):
        return self.name

    @name.setter
    def name(self, name):
        self.__name = name

    def getName(self):
        return self.__name

    def setName(self, name):
        self.__name = name


#
class Employee(Person):
    def __init__(self, name, empnum):
        print('Employee.__init__()')
        self.__empnum = ''
        #super().__init__(name)
        self.empnum = empnum

    def __del__(self):
        print('Employee.__del__()')

    @property
    def empnum(self):
        return self.__empnum

    @property
    def details(self):
        return self.get_employee_details()

    @empnum.setter
    def empnum(self, empnum):
        self.__empnum = empnum

    def get_employee_details(self):
        x = self.name + ':::' + self.empnum
        return x
