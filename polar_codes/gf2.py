"""GF(2) 矩阵工具"""
import numpy as np


def F2(n):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F
    for _ in range(n - 1):
        G = np.kron(G, F)
    return G


def gf2_inv(A):
    A = A.copy() % 2
    n = A.shape[0]
    I = np.eye(n, dtype=int)
    for col in range(n):
        if A[col, col] == 0:
            swap = next(i for i in range(col + 1, n) if A[i, col])
            A[[col, swap]] = A[[swap, col]]
            I[[col, swap]] = I[[swap, col]]
        for r in range(n):
            if r != col and A[r, col]:
                A[r] ^= A[col]
                I[r] ^= I[col]
    return I % 2
