import numpy as np
def normalize(a):
    a=np.mat(a)
    m,n=np.shape(a)
    for i in range(0,n):
        max=np.max(a[:,i])
        min=np.min(a[:,i])
        for j in range(0,m):
            a[j,i]=(a[j,i]-min)/(max-min)
    return np.array(a)
