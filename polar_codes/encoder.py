"""
极化码编码器
编码：x = u F^{⊗ n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 perm，满足 out[i] = in[perm[i]]"""
    n = int(np.log2(N))
    perm = np.arange(N, dtype=int)
    for i in range(N):
        perm[i] = bit_reverse_int(i, n)
    return perm


def bit_reverse_int(i, n):
    rev = 0
    x = i
    for _ in range(n):
        rev = (rev << 1) | (x & 1)
        x >>= 1
    return rev


def polar_encode(u):
    """
    极化码编码：x = u F^{⊗ n}（蝶形结构，与 SCD 文献一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u.astype(int)


def polar_encode_with_br(u):
    """蝶形后附加比特倒序（部分教材定义 G_N = B_N F^{⊗ n}）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(u))]


def polar_encode_matrix(u):
    """矩阵法编码（用于校验）：x = u @ (B_N F^{⊗ n}) mod 2"""
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)

    def gn(m):
        if m == 0:
            return np.array([[1]], dtype=np.int8)
        return np.kron(F, gn(m - 1))

    G = gn(n)
    perm = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, p in enumerate(perm):
        B[i, p] = 1
    GN = (B @ G) % 2
    return (u @ GN) % 2
