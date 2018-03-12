import numpy as np

def transpose(x):
    l = np.size(x[:,0])
    c = np.size(x[0,:])
    z = np.zeros([c,l])

    for i in range(0,l):
        for j in range(0,c):
            z[j,i]=x[i,j]
    return z

def norme(v,n):
    s=0
    for i in range(n):
        s+= v[i]**2
    return s

a=np.matrix([[1,2],[1,2]])
print(a)
b=transpose(a)
print(b)

def householder(x,y):
    n=len(x)
    u=(x-y)/(np.linalg.norm(x)-np.linalg.norm(y))
    
