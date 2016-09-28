__author__ = 'Praveenb'


class GS:
    def __init__(self, x):
        print('GS.__init__()')
        self.attribute_x = x

    def __del__(self):
        print('GS.__del__()')

    @property
    def attribute_x(self):
        print('@property : attribute_x(self)')
        return self.__x

    @property
    def condition(self):
        print('@property : condition')
        return self.__x

    @attribute_x.setter
    def attribute_x(self, x):
        print('@attribut_x.setter : attribute_x(self, x)')
        if (x < 0):
            self.__x = 0
        elif (x > 1000):
            self.__x = 1000
        else:
            self.__x = x
