import numpy as np

def single_classification(I, D):
    return np.sign(np.sum([D[i,0] for i in I])) #davor D[i], das war aber glaube ich falsch,da wir ja nur über die 
    #y_i aufsummieren wollen und nicht über die x-Werte
