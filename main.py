from classify import classify
import numpy as np
import time

tic = time.time()
classify("bananas-1-2d", np.linspace(1, 200, 200), 5)
toc = time.time()
print("%.10f seconds" % (toc - tic))
