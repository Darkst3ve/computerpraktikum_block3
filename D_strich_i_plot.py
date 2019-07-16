import numpy as np
import matplotlib.pyplot as plt
from classify import classify
from file_import import file_import
from nearest_points import nearest_points_naive_l1
from nearest_points import nearest_points_naive_sup

train = file_import("bananas-2-2d.train.csv")
test = file_import("bananas-2-2d.test.csv")
test_index = np.int(np.random.rand() * len(test))
k = 100
l = 5
n = train.shape[0]  # Anzahl Punkte
m = train.shape[1]  # Anzahl Dimensionen; Beachte: Enthält die Klassifikation
block_size = n // l  # Größe der D_i
D_i_array = np.zeros((l, block_size, m))  # Zu untersuchende Punkte von train
D_strich_i_array = np.zeros((l, block_size * (l - 1), m))  # Zu vergleichende Punkte von train
for i in range(l):
    # Erzeuge alle benötigten Arrays an Punkten, i wird in der ersten Koordinate dieser Arrays indiziert
    D_i_array[i] = train[i * block_size:(i + 1) * block_size, :]
    lower_points = train[0:i * block_size, :]
    upper_points = train[(i + 1) * block_size:l * block_size, :]
    D_strich_i_array[i] = np.vstack((lower_points, upper_points))

for i in range(l):
    plt.figure(i)
    nearest = nearest_points_naive_sup(test[test_index, :], D_strich_i_array[i, :, :], k)
    rest = [i for i in range((l - 1) * block_size) if i not in nearest]
    plt.scatter(D_strich_i_array[i, nearest, 1], D_strich_i_array[i, nearest, 2], s=0.6, c='r')
    plt.scatter(D_strich_i_array[i, rest, 1], D_strich_i_array[i, rest, 2], s=0.6, c='k')
    plt.scatter(test[test_index, 1], test[test_index, 2], s=1.5, c='b')
plt.show()
