import numpy as np
from file_import import file_import
from nearest_points import nearest_points_naive_sup
from nearest_points import nearest_points_naive_l1
from nearest_points import nearest_points_opt_l1
from nearest_points import nearest_points_opt_sup
import time
import csv


def classify(file_name, KSET, l):
    tic = time.time()
    k_max = max(KSET)
    test = file_import(file_name + ".test.csv")  # Vollständiges Array; Enthält Klassifikation
    train = file_import(file_name + ".train.csv")  # dito
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
    print("Initialisierung : %.10f seconds" % (toc - tic))
    tic = time.time()
    for i in range(l):
        # Bestimme die k_max nächsten Nachbarn
        for j in range(0, block_size):
            index_array[block_size * i + j, :] = nearest_points_opt_l1(D_i_array[i, j, :], D_strich_i_array[i, :, :], k_max)  # sic
    list_ks = []
    toc = time.time()
    print("Nächste Nachbarn in train : %.10f seconds" % (toc - tic))
    tic = time.time()
    new_array = np.zeros((l, block_size, k_max))  # Enthält Summen der Klassifikationen (ohne Signum) der n Punkte zu allen nächsten Nachbarn (bis k_max)
    for i in range(l):
        for j in range(block_size):
            new_array[i, j, :] = np.cumsum(D_strich_i_array[i, index_array[i * block_size + j, :], 0])
#            new_array[i, j, 0] = np.sum(D_strich_i_array[i, index_array[i * block_size + j, 0], 0])
#            for k in range(len(KSET) - 1):
#                new_array[i, j, k + 1] = new_array[i, j, k] + D_strich_i_array[i, index_array[i * block_size + j, k + 1], 0]
    temp1_array = np.sign(new_array)
    temp2_array = np.zeros((l, block_size, k_max))
    for i in range(l):
        for j in range(block_size):
            for k in range(len(KSET)):
                if temp1_array[i, j, k] == 0:
                    temp1_array[i, j, k] = 1
                if D_i_array[i, j, 0] == temp1_array[i, j, k]:
                    temp2_array[i, j, k] = 0
                else:
                    temp2_array[i, j, k] = 1
    temp3_array = np.sum(temp2_array, 1) / block_size
    temp4_array = np.sum(temp3_array, 0) / l
#    print(temp4_array)  
    toc = time.time()
    print("Bestimmung von k_stern : %.10f seconds" % (toc - tic))
    k_stern = np.argmin(temp4_array)
    print("k_stern = " + str(k_stern))
    print("Klassifikationsfehlerrate: " + str(temp4_array[k_stern]))
    o = len(test)
    test_classification = np.zeros(o)
    test_index_array = np.zeros((l, o, k_stern), dtype=int)
    tic = time.time()
    for i in range(l):
        for j in range(o):
            test_index_array[i, j, :] = nearest_points_opt_l1(test[j, :], D_strich_i_array[i, :, :], k_stern)
    toc = time.time()
    print("Nächste Nachbarn von test : %.10f seconds" % (toc - tic))
    tic = time.time()
    for j in range(o):
        temp1 = 0
        for i in range(l):
            temp2 = np.sign(np.sum(D_strich_i_array[i, test_index_array[i, j, :], 0]))
            temp1 += temp2
            if temp2 == 0:
                temp1 += 1
        test_classification[j] = np.sign(temp1) # Empirisch : Wird nicht Null, also keine weitere Abfrage nötig
    toc = time.time()
    print("Bestimmung der Klassifikation : %.10f seconds" % (toc - tic))
#    print(test_classification)
    test[:, 0] = test_classification
    with open(file_name + ".result.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test)
#    print(test)
    return test
