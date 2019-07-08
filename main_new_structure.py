import numpy as np
from file_import import file_import
from nearest_points import nearest_points_naive_sup_2
from nearest_points import nearest_points_naive_l1
from single_classification import single_classification
import time

def classify(nametest,nametrain, Kset, l):
    test = file_import(nametest)
    train= file_import(nametrain)
    D_test = test[:,1:]
    D_train= train[:,1:]
    n=len(train)
    großk = int(max(Kset))
    index_array=np.zeros((n,großk), dtype=int)
    m_i=0
    tic = time.time()
    for i in range(l):
            block_size = n//l
            train_i= train[i*block_size:(i+1)*block_size,:]
            a_i= train[0:i*block_size,:]
            b_i= train[(i+1)*block_size:,:]
            train_strich_i = np.vstack((a_i,b_i))
            D_train_i = train_i[:,1:] #erste Spalte nicht dran
            c_i= D_train[0:i*block_size,:]
            d_i= D_train[(i+1)*block_size:,:]
            D_train_strich_i = np.vstack((c_i,d_i))
            index_array_i=np.zeros((m_i,großk),dtype=int)
            for j in range(0,len(D_train_i)):
                index_j_i= nearest_points_naive_sup_2(D_train_i[j,:],D_train_strich_i,großk)
                index_array[m_i+j,:]=index_j_i
            m_i= m_i + len(D_train_i)
    toc = time.time()
    print("%.10f seconds" % (toc - tic))
    list_ks = []
    tic = time.time()
    for k in Kset:
        errorarray=[]
        for i in range(l):
            m_i= len(D_train_i)
            block_size = n//l
            train_i= train[i*block_size:(i+1)*block_size,:]
            a_i= train[0:i*block_size,:]
            b_i= train[(i+1)*block_size:,:]
            train_strich_i = np.vstack((a_i,b_i))
            D_train_i = train_i[:,1:] #erste Spalte nicht dran
            c_i= D_train[0:i*block_size,:]
            d_i= D_train[(i+1)*block_size:,:]
            D_train_strich_i = np.vstack((c_i,d_i))
            index_array_i=np.zeros((m_i,großk),dtype=int)
            C_i=[]
            for j in range(0,m_i):
                if train_i[j,0] == np.sign(np.sum(train_strich_i[index_array[m_i + j,:int(k)],0])):
                    c=0
                else:
                    c=1
                C_i.append(c)
            m_i += len(D_train_i)
            error_classification_i= 1/m_i *sum(C_i)
            errorarray.append(error_classification_i)
        middle_k= (1/l)*sum(errorarray)
        list_ks.append(middle_k)
    toc = time.time()
    print("%.10f seconds" % (toc - tic))
    print(list_ks)
    print([np.abs(list_ks[i] - list_ks[i + 1]) for i in range(len(list_ks) - 1)])
    print(list_ks.index(min(list_ks)))
