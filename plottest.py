import matplotlib.pyplot as plt
from classify import classify
from file_import import file_import 
import numpy as np
set = classify("bananas-2-2d",  np.arange(1,200),  5)
#set=file_import("bananas-2-2d.train.csv")
list_1=[i for i in range(len(set)) if set[i,0]==1]
list_2=[i for i in range(len(set)) if set[i,0]==-1]
plt.scatter(set[list_1, 1], set[list_1, 2], s=0.6, c="k", label="1")
plt.scatter(set[list_2, 1], set[list_2, 2], s=0.6, c="r", label="-1")
plt.title("bananas-2-2d.train")
plt.legend(markerscale =7.5, title="Klassifikation")
plt.show()
#plt.savefig("bananas-2-2d.train.pdf", bbox_inches='tight')


