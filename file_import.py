import numpy as np
import csv

def file_import(filename):
    with open(filename, newline='') as csvfile:
        A = np.array(list(csv.reader(csvfile, skipinitialspace=True, delimiter=','))).astype(np.float)
        return A

