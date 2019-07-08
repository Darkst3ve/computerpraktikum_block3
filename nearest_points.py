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

def nearest_points_naive_sup_2(x, D, k):
    n = D.shape[0]
    if k > n:
        k = n  # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
    E = np.max(np.abs(x - D), 1)
    I = np.argsort(E)
    return I[:k]


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


def nearest_points_tryhard(x, D, k, sort_coordinate=0):
    n = D.shape[0]
    D = D[:, 1:]
    x = x[1:]
    if k > n:
        k = n  # Versuche nicht mehr nächste Punkte zu finden als es insgesamt gibt, sollte normalerweise nicht auftreten
        warnings.warn("Anzahl gesuchter nächster Punkte ist größer als Anzahl verfügbarer Punkte")
    
    I = np.argsort(D[:, sort_coordinate])
    closest_found = False
    closest_in_sorted = n // 2
    step = n // 4
    while(not closest_found):
        compare = x[sort_coordinate] - D[I[closest_in_sorted], sort_coordinate]
        if compare == 0:
            closest_found = True
        elif compare < 0:
            closest_in_sorted -= step
            step *= 1 // 2
            if step < 1 // 2:
                closest_found = True
        elif compare > 0:
            closest_in_sorted += step
            step *= 1 // 2
            if step < 1 // 2:
                closest_found = True
    # Get ready:
    temp = [closest_in_sorted + ((-1) ** (1 + i // 2)) * i // 2 for i in range(1, 2 * n) if closest_in_sorted + ((-1) ** (1 + i // 2)) * i // 2 >= 0 and closest_in_sorted + ((-1) ** (1 + i // 2)) * i // 2 < n]
    closest_values = [-np.max(np.abs(x - D[I[closest_in_sorted]]))]
    for i in range(len(temp)):
        hp.heapify(closest_values)
        next_value = -np.max(np.abs(x - D[I[temp[i]]]))
        if len(closest_values) < k:
            hp.heappush(closest_values, next_value)
            continue
        if closest_values[0] < next_value:
            hp.heapreplace(closest_values, next_value)
            continue
        if closest_values[0] > np.abs(x[sort_coordinate] - D[temp[i]][sort_coordinate]):
            break
    
    temp_range = range(i)
    T = dict(zip(list([-np.max(np.abs(x - D[I[j]])) for j in temp_range]), list(I[temp_range])))
    return np.array([T[i] for i in closest_values])


x = np.array([1, 2, 3, 4])
D = np.array([[1, 1, 1, 1.1], [1, 1, 1, 2.2], [1, 2, 3, 6.3], [0, 2, 4, 5.4], [2, 3, 4, 5.5]])
# print(nearest_points_naive_l1(x, D, 2))
# print(nearest_points_heap_l1(x, D, 2))
# print(nearest_points_naive_sup(x, D, 2))
# print(nearest_points_heap_sup(x, D, 2))
# print(nearest_points_tryhard(x, D, 2))

# temp = np.random.rand(1000000, 4)
# A = nearest_points_naive_sup([0.5, 0.5, 0.5, 0.5], temp, 100)
# B = nearest_points_heap_sup([0.5, 0.5, 0.5, 0.5], temp, 100)
# print([item for item in list(A) + list(B) if item not in set(A).intersection(B)]) # Soll [] returnen
 
