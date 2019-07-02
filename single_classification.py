import numpy as np

def single_classification(I, D):
    return np.sign(np.sum([D[i] for i in I]))