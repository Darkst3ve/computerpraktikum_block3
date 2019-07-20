import numpy as np
import time
from file_import import file_import
from nearest_points import nearest_points_naive_l1
from nearest_points import nearest_points_naive_sup
from scipy.spatial import KDTree

file_name = "toy-10d.train.csv"

data = file_import(file_name)
n = data.shape[0]
result_array_1 = np.zeros((n, 200))
tic = time.time()
for i in range(n):
    result_array_1[i, :] = nearest_points_naive_l1(data[i, :], data, 200)
toc = time.time()
print("l1-naiv : %.10f seconds" % (toc - tic))

data = file_import(file_name)
n = data.shape[0]
result_array_2 = np.zeros((n, 200))
tic = time.time()
for i in range(n):
    result_array_2[i, :] = nearest_points_naive_sup(data[i, :], data, 200)
toc = time.time()
print("lsup-naiv : %.10f seconds" % (toc - tic))

data = file_import(file_name)
n = data.shape[0]
result_array_3 = np.zeros((n, 200))
tic = time.time()
kdt = KDTree(data[:, 1:])
for i in range(n):
    result_array_3[i, :] = kdt.query(data[i, 1:], k=200, p=1)[1]
toc = time.time()
print("l1-kdt : %.10f seconds" % (toc - tic))

data = file_import(file_name)
n = data.shape[0]
result_array_4 = np.zeros((n, 200))
tic = time.time()
kdt = KDTree(data[:, 1:])
for i in range(n):
    result_array_4[i, :] = kdt.query(data[i, 1:], k=200, p=float("inf"))[1]
toc = time.time()
print("lsup-kdt : %.10f seconds" % (toc - tic))
