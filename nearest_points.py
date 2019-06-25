import numpy as np
import warnings

def nearest_points_naive_l1(x, D, k):
    n = D.shape[0]
    if k > n:
        k = n # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
    E = np.zeros(n)
    for i in range(n):
        E[i] = np.sum(np.abs(x - D[i]))
    I = np.argsort(E)
    return I[:k]

def nearest_points_naive_sup(x, D, k):
    n = D.shape[0]
    if k > n:
        k = n # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
    E = np.zeros(n)
    for i in range(n):
        E[i] = np.max(np.abs(x - D[i]))
    I = np.argsort(E)
    return I[:k]

x = np.array([1, 2, 3, 4])
D = np.matrix([[1, 1, 1, 1],[1, 1, 1, 2], [1, 2, 3, 6], [0, 2, 4, 5], [2, 3, 4, 5]])
print(nearest_points_naive_l1(x, D, 2))
print(nearest_points_naive_sup(x, D, 2))