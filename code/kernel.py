import numpy as np
import numpy.linalg as la

class Kernel(object):
    """Implements list of kernels from http://en.wikipedia.org/wiki/Support_vector_machine
    Reference from https://github.com/ajtulloch/svmpy
    """

    @staticmethod
    def linear(x1, x2):
        return np.dot(x1, x2)

    @staticmethod
    def gaussian(x, y, sigma = 0.5):
        return np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))

    @staticmethod
    def polykernel(x, y, dimension = 3, offset = 1):
        return (offset + np.dot(x, y)) ** dimension

