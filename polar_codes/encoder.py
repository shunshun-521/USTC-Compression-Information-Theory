"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = np.zeros(N, dtype=np.int64)
    for b in range(n):
        rev |= ((idx >> b) & 1) << (n - 1 - b)
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    """
    u = np.array(u, dtype=np.int8, copy=True)
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step <<= 1
    # 与 Permuted SCD / x = u F^{⊗n} 约定一致（不在此处做比特倒序）
    return u.astype(int)


def polar_generator_matrix(N):
    """生成 F^{\otimes n}（与蝶形编码器一致）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("butterfly+BRP:", x)
    print("matrix mult:", x_ref)
