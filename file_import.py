import numpy as np
import csv

A = []

with open('bananas-1-2d.test.csv', newline='') as csvfile:
    A = np.array(list(csv.reader(csvfile, skipinitialspace=True, delimiter=','))).astype(np.float)

print(A)
print(type(A))
print(type(A[0]))
print(type(A[0][0]))