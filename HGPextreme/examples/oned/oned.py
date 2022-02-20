import numpy as np


def S(x):
    # 1D random function
    return (f(x) + np.random.normal(0, r(x))).reshape(-1)

def f(x):
    # mean function
    return (x-5)**2

def r(x):
    # std function
    return 0.1 + 0.1 * x ** 2