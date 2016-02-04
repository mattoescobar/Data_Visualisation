import numpy as np
import scipy as sp


class GTM(object):

    def __init__(self, input_variable=sp.rand(100, 1)):
        self.input_variable = input_variable

test = GTM()
print test.input_variable
