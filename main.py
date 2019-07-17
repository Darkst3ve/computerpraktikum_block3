from classify import classify
import numpy as np
import time

tic = time.time()
classify("bananas-2-2d", np.arange(1, 200), 5)
toc = time.time()
print("Gesamtlaufzeit:")
print("%.10f seconds" % (toc - tic))
