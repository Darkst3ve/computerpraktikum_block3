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

plt.figure(1)
nearest = nearest_points_naive_sup(test[test_index, :], train, k)
rest = [i for i in range(len(train)) if i not in nearest]
plt.scatter(train[nearest, 1], train[nearest, 2], s=0.6, c='r')
plt.scatter(train[rest, 1], train[rest, 2], s=0.6, c='k')
plt.scatter(test[test_index, 1], test[test_index, 2], s=1.5, c='b')

plt.figure(2)
nearest = nearest_points_naive_l1(test[test_index, :], train, k)
rest = [i for i in range(len(train)) if i not in nearest]

plt.scatter(train[nearest, 1], train[nearest, 2], s=0.6, c='r')
plt.scatter(train[rest, 1], train[rest, 2], s=0.6, c='k')
plt.scatter(test[test_index, 1], test[test_index, 2], s=1.5, c='b')
plt.show()

print(nearest)
print(train[nearest, :])
print(sum(train[nearest, 0]))
