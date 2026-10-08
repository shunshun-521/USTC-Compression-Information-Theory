"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_f_transform(u):
    """
    非系统化极化编码：u -> x（与 mcba1n / Arikan 蝶形一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u.astype(int)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = B_N * F^{\\otimes n} * u
    """
    u = np.asarray(u, dtype=np.int8)
    x = polar_f_transform(u)
    br = bit_reversal_permutation(len(u))
    return x[br].astype(int)


def build_generator_matrix(N):
    """构建 G_N = B_N F^{\\otimes n}（用于测试）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    G = build_generator_matrix(4)
    print("matrix multiply:", (u @ G) % 2)
