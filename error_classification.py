import numpy as np
from single_classification import single_classification
from nearest_points import nearest_points_naive_sup
def error_classification(A, B, k):
    size= np.shape(A)   
    m=size[0] #Zeilenanzahl
    d=size[1]-1 #Dimension der Daten
    C=[]
    for i in range(0,m): #für jede Zeile der Daten
        if A[i,0]== single_classification(nearest_points_naive_sup(B[i,1:], B[:,1:], k), B): 
            c= 0 
        else:
            c= 1 
        C.append(c)
    result = (1/m)*(np.sum(C)) #Berechnen der Funktion
    return result
