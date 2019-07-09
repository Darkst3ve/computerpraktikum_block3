import numpy as np
from file_import import file_import
from nearest_points import nearest_points_naive_sup
import time
import warnings


def classify(name, KSET, l):
    tic = time.time()
    k_max = max(KSET)
    test = file_import(name + ".test.csv")  # Vollständiges Array; Enthält Klassifikation
    train = file_import(name + ".train.csv")  # dito
    n = train.shape[0]  # Anzahl Punkte
    m = train.shape[1]  # Anzahl Dimensionen; Beachte: Enthält die Klassifikation
    index_array = np.zeros((n, k_max), dtype=int)  # Enthält alle Indizes der k_max nächsten Nachbarn aller Punkte
    block_size = n // l  # Größe der D_i
    D_i_array = np.zeros((l, block_size, m))  # Zu untersuchende Punkte von train
    D_strich_i_array = np.zeros((l, block_size * (l - 1), m))  # Zu vergleichende Punkte von train
    for i in range(l):
        # Erzeuge alle benötigten Arrays an Punkten, i wird in der ersten Koordinate dieser Arrays indiziert
        D_i_array[i] = train[i * block_size:(i + 1) * block_size, :]
        lower_points = train[0:i * block_size, :]
        upper_points = train[(i + 1) * block_size:l * block_size, :]
        D_strich_i_array[i] = np.vstack((lower_points, upper_points))
    toc = time.time()
    print("%.10f seconds" % (toc - tic))
    tic = time.time()
    
    for i in range(l):
        # Bestimme die k_max nächsten Nachbarn
        for j in range(0, block_size):
            index_array[block_size * i + j, :] = nearest_points_naive_sup(D_i_array[i, j, :], D_strich_i_array[i, :, :], k_max)  # sic
    list_ks = []
    toc = time.time()
    print("%.10f seconds" % (toc - tic))
    tic = time.time()
    for k in KSET:
        errorarray = np.zeros(l)
        for i in range(l):
            C_i = np.zeros(block_size)
            for j in range(0, block_size):
                temp = np.sign(np.sum(D_strich_i_array[i, index_array[i * block_size + j, :k], 0]))
                if temp == 0:
                    temp = 1
                if D_i_array[i, j, 0] == temp:
#                if D_i_array[i, j, 0] == single_classification(index_array[i * block_size + j, :int(k)], D_strich_i_array[i, :, :]):
                    c = 0
                else:
                    c = 1
                C_i[j] = c
            errorarray[i] = sum(C_i) / block_size
        middle_k = (1 / l) * sum(errorarray)
        list_ks.append(middle_k)
    toc = time.time()
    print("%.10f seconds" % (toc - tic))
    print(list_ks)
#    print([np.abs(list_ks[i] - list_ks[i + 1]) for i in range(len(list_ks) - 1)])
    k_stern = np.int(list_ks.index(min(list_ks)))
    print(k_stern)
    o = len(test)
    test_classification = np.zeros(o)
    test_index_array = np.zeros((l, o, k_stern), dtype = int)
    tic = time.time()
    for i in range(l):
        for j in range(o):
            test_index_array[i, j, :] = nearest_points_naive_sup(train[j, :], D_strich_i_array[i, :, :], k_stern)
    toc = time.time()
    print("%.10f seconds" % (toc - tic))
    tic = time.time()
    for j in range(o):
        temp = 0
        for i in range(l):
            temp += np.sign(np.sum(D_strich_i_array[i, test_index_array[i, j, :]]))
            if np.sign(np.sum(D_strich_i_array[i, test_index_array[i, j, :]])) == 0:
                temp += 1
        test_classification[j] = np.sign(temp)
#        if test_classification[j] == 0:
#            test_classification[j] = 1
#            warnings.warn("sign returns 0")
    toc = time.time()
    print("%.10f seconds" % (toc - tic))
    print(test_classification)
