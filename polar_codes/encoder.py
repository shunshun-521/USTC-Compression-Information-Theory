"""
极化码编码器
编码利用蝶形递归结构，复杂度 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for j in range(n):
            r = (r << 1) | ((i >> j) & 1)
        rev[i] = r
    return rev


def _butterfly_combine(a, b):
    """极化蝶形合并：[a XOR b, b]"""
    out = np.zeros(len(a) + len(b), dtype=np.int8)
    out[:len(a)] = (a ^ b)
    out[len(a):] = b
    return out


def polar_encode(u):
    """
    极化码编码（蝶形递归结构）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N)) + 1
    m = 1
    for _ in range(n - 1):
        for i in range(0, N, 2 * m):
            u[i:i + 2 * m] = _butterfly_combine(u[i:i + m], u[i + m:i + 2 * m])
        m *= 2
    return u.astype(int)


def polar_encode_matrix(u):
    """基于生成矩阵 u @ (B @ F^N) 的编码（验证用）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))

    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)

    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1

    return (u @ (B @ G)) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print(f"u={u} -> x={polar_encode(u)}")
