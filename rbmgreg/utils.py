import numpy as np


def isunique(lst): return len(set(lst))==1

def sigmoid(x):
    # from peter's rbm
    return 1.0 / (1.0 + np.exp(-1.0 * x))
