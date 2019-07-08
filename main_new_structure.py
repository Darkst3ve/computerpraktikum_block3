import numpy as np
from file_import import file_import
from nearest_points import nearest_points_naive_sup
def classify(nametest,nametrain, Kset, l):
    test = file_import(nametest)
    train= file_import(nametrain)
    D_test = test[:,1:]
    D_train= train[:,1:]
    n=len(train)
    großk = int(max(Kset))
    for i in range(l):
            train_i= train[(i*n)//l:((i+1)*n)//l,:]
            a_i= train[0:(i*n)//l,:]
            b_i= train[(i+1)*n//l:,:]
            train_strich_i = np.vstack((a_i,b_i))
            D_train_i = train_i[:,1:] #erste Spalte nicht dran
            c_i= D_train[0:(i*n)//l,:]
            d_i= D_train[(i+1)*n//l:,:]
            D_train_strich_i = np.vstack((c_i,d_i))
            m_i= len(D_train_i)
            for j in range(0,m_i):
                    index_j_i= nearest_points_naive_sup(D_train_strich_i[j,:],D_train_strich_i,großk)
    list_ks = []
    for k in Kset:
        errorarray=[]
        for i in range(l):
            m_i= len(D_train_i)
            C_i=[]
           
            for j in range(0,m_i):
                if train[j,0] == np.sign(np.sum(train_strich_i[index_j_i[:int(k)],0])):
                    c=0
                else:
                    c=1
                C_i.append(c)
            error_classification_i= 1/m_i *sum(C_i)
            errorarray.append(error_classification_i)
        middle_k= (1/l)*sum(errorarray)
        list_ks.append(middle_k)
    print(list_ks)
    print(list_ks.index(min(list_ks))
