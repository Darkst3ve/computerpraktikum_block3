from classify import classify
import numpy as np
import time

tic = time.time()
classify("toy-10d", np.arange(1, 200), 5)
toc = time.time()
print("%.10f seconds" % (toc - tic))
