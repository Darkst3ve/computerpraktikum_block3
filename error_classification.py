import numpy as np
from file_functionD import functionD
#functionD ist die Funktion die Marcus programmiert, das kann man gegebenenfalls noch umbenennnen
def error_classification(A):  
    size= np.shape(A)   
    m=size[0] #Zeilenanzahl
    d=size[1]-1 #Dimension der Daten
    C=[]
    for i in range(0,m): #für jede Zeile der Daten
        if A[i,0]== functionD(A[i,1:d+1]): 
            c= 0 
        else:
            c= 1 
        #dieser Abschnitt dient zur berechnung der charakteristischen Funktion
    C.append(c)
    result = (1/m)*(np.sum(C)) #Berechnen der Funktion
    return print(result)
