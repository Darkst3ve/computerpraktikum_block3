from file_import import file_import
from nearest_points import nearest_points_naive_sup
from single_classification import single_classification
from error_classification import error_classification
import numpy as np

A = file_import("bananas-1-2d.test.csv")
D = A[:,1:]
n = len(D)
k = 5
l = 5
error_array = np.zeros(5)
for i in range(l):
    A_i= A[(i*n)//l:((i+1)*n)//l,:] #erste Spalte dran
    D_i = D[(i*n)//l:((i+1)*n)//l,:] #erste Spalte nicht dran
    D_strich_i = np.array([element for element in D if element not in D_i])
    #Komplemente bestimmen das ist noch NICHT richtig so, da manche Elemente doppelt vorkommen und diese dann NICHT
    #im Komplement sind, Also die D_strich_i sind noch zu klein(es wird zu viel gelöscht)
    error_array[i] = error_classification(A_i, D_strich_i, k)
 
print(error_array)
