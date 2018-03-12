# Compression d'image

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

img_full = mp.image.imread("p3_takeoff_base.png")
print(img_full[250][250])

def transform(x) :
    n = np.shape(x)[0]
    M = np.zeros((n,n))

    for i in range(n):
        M[i][i] = x[i]
    return M

def trunc(M):
    (n,m) = np.shape(M)
    for i in range(n) :
        for j in range(m) :
            if M[i][j] < 0 :
                M[i][j] = 0
            elif M[i][j] > 255 :
                M[i][j] = 255
    return M

def split(M):
    (m,n,p) = np.shape(M)
    print("M[250][250] = ", M[250][250])
    R = np.zeros((m, n))
    G = np.zeros((m, n))
    B = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            R[i][j] = M[i][j][0]
            G[i][j] = M[i][j][1]
            B[i][j] = M[i][j][2]
    print("R[200][200] = ", R[250][250])
    print("G[200][200] = ", G[250][250])
    print("B[200][200] = ", B[250][250])
    
    return (R, G, B)

def merge(R, G, B):
    (m,n) = np.shape(R)
    M = np.zeros((m,n,3))
    for i in range(m) :
        for j in range(n) :
            M[i][j] = [R[i][j], G[i][j], B[i][j]]
    return M

            
def compress_one(M, k) :
    n = np.shape(M)[0]
    
    for i in range(k + 1,n):
        M[i][i] = 0

def compress(M, k):
    (R, G, B) = split(M)
    (Ur, Sr, Vr) = np.linalg.svd(R)
    (Ug, Sg, Vg) = np.linalg.svd(G)
    (Ub, Sb, Vb) = np.linalg.svd(B)

    Sr = transform(Sr)
    Sg = transform(Sg)
    Sb = transform(Sb)
    
    compress_one(Sr, k)
    compress_one(Sg, k)
    compress_one(Sb, k)

    print(Sr[250][250])
    print(Sg[250][250])
    print(Sb[250][250])
    
    R = trunc(np.dot(Ur, np.dot(Sr, Vr)))
    G = trunc(np.dot(Ug, np.dot(Sg, Vg)))
    B = trunc(np.dot(Ub, np.dot(Sb, Vb)))

    
    Mcomp = merge(R, G, B)
    return Mcomp

    
img_10 = compress(img_full, 10)

plt.figure("Image")
plt.imshow(img_10)
plt.show()
