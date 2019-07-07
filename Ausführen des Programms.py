from main_new_structure import classify 
import numpy as np
import time
start_time = time.time()
classify("bananas_1_4d.test.csv","bananas_1_4d.train.csv",  np.linspace(1,200,200),  5)
end_time = time.time()
print("%.10f seconds" % (end_time - start_time))