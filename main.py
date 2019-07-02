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
    D_i = D[,:]
    D_strich_i = np.array([element for element in D if element not in D_i])
    error_array[i] = error_classification(D_i, D_strich_i, k)

print(error_array)