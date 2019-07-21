import numpy as np
import warnings
import heapq as hp
import time

'''
Alle enthaltenen Funktionen sind konstruiert so dass die Eingabe D das gesamte zu vergleichende Punktarray ist,
also mit der Klassifikation in der ersten Spalte enthalten. Diese wird dann jeweils zur Verarbeitung gestrichen
'''


def nearest_points_naive_l1(x, D, k):
    n = D.shape[0]
    D = D[:, 1:]
    x = x[1:]
    if k > n:
        k = n  # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
    E = np.sum(np.abs(x - D), 1)
    I = np.argsort(E)
    return I[:k]


def nearest_points_naive_sup(x, D, k):
    n = D.shape[0]
    D = D[:, 1:]
    x = x[1:]
    if k > n:
        k = n  # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
    E = np.max(np.abs(x - D), 1)
    I = np.argsort(E)
    return I[:k]

def nearest_points_opt_l1(x, D, k):
    n = D.shape[0]
    D = D[:, 1:]
    x = x[1:]
    if k > n:
        k = n  # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
    E = np.sum(np.abs(x - D), 1)
    P = np.argpartition(E, k)[:k]
    map = dict(zip(E[P], P))
    I = np.sort(E[P])
    return np.array([map[i] for i in I])



def nearest_points_opt_sup(x, D, k):
    n = D.shape[0]
    D = D[:, 1:]
    x = x[1:]
    if k > n:
        k = n  # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
    E = np.max(np.abs(x - D), 1)
    if k == 0:
        return np.argmin(E)
    P = np.argpartition(E, k)[:k]
    map = dict(zip(E[P], P))
    I = np.sort(E[P])
    return np.array([map[i] for i in I])


def nearest_points_heap_sup(x, D, k):
    n = D.shape[0]
    D = D[:, 1:]
    x = x[1:]
    if k > n:
        k = n  # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
#    E = np.zeros(n)
    E = np.max(np.abs(x - D), 1)
    
    H = list(-E[0:k])
    hp.heapify(H)
    tic = time.time()
    for i in range(k, n):
        if -E[i] < H[0]:
            continue 
        hp.heapreplace(H, -E[i])
    
    toc = time.time()
    T = dict(zip(E, range(n)))
    H = list(-np.array(H))
    print(toc - tic)
    return np.array([T[i] for i in H])


def nearest_points_heap_l1(x, D, k):
    n = D.shape[0]
    D = D[:, 1:]
    x = x[1:]
    if k > n:
        k = n  # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
    E = np.sum(np.abs(x - D), 1)
    
    H = list(-E[0:k])
    hp.heapify(H)
    for i in range(k, n):
        if -E[i] < H[0]:
            continue 
        hp.heapreplace(H, -E[i])
    
    T = dict(zip(E, range(n)))
    H = list(-np.array(H))
    return np.array([T[i] for i in H])




x = np.array([1, 2, 3, 4])
D = np.array([[1, 1, 1, 1.1], [1, 1, 1, 2.2], [1, 2, 3, 6.3], [0, 2, 4, 5.4], [2, 3, 4, 5.5]])
# print(nearest_points_naive_l1(x, D, 2))
# print(nearest_points_heap_l1(x, D, 2))
# print(nearest_points_naive_sup(x, D, 2))
# print(nearest_points_heap_sup(x, D, 2))

# temp = np.random.rand(1000000, 4)
# A = nearest_points_naive_sup([0.5, 0.5, 0.5, 0.5], temp, 100)
# B = nearest_points_heap_sup([0.5, 0.5, 0.5, 0.5], temp, 100)
# print([item for item in list(A) + list(B) if item not in set(A).intersection(B)]) # Soll [] returnen
 
