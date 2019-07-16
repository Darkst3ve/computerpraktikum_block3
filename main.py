from classify import classify
import numpy as np
import time

tic = time.time()
classify("bananas-2-2d", np.linspace(1, 200, 200, dtype = int), 5)
toc = time.time()
print("%.10f seconds" % (toc - tic))
