import numpy as np
from single_classification import single_classification
from nearest_points import nearest_points_naive_sup
#functionD ist die Funktion die Marcus programmiert, das kann man gegebenenfalls noch umbenennnen
def error_classification(A, D, k):
    size= np.shape(A)   
    m=size[0] #Zeilenanzahl
    d=size[1]-1 #Dimension der Daten
    C=[]
    for i in range(0,m): #für jede Zeile der Daten
        print(i)
        print(A[i, 0])
        print(single_classification(nearest_points_naive_sup(A[i], D, k), D))
        if A[i,0]== single_classification(nearest_points_naive_sup(A[i], D, k), D): 
            c= 0 
        else:
            c= 1 
        #dieser Abschnitt dient zur berechnung der charakteristischen Funktion
        C.append(c)
    result = (1/m)*(np.sum(C)) #Berechnen der Funktion
    return result
