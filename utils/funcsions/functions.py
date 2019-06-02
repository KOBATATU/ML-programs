import numpy as np


class Sigmoid():
    def __call__(self,x):
        return 1/(1 + np.exp(-x))