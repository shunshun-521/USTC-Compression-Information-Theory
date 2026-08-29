"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def _bit_reversal_indices(N):
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    return _bit_reversal_indices(N)


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))

    for layer in range(n):
        step = 2 ** layer
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]

    br = _bit_reversal_indices(N)
    return u[br].astype(int)


def verify_encoder(N=4):
    """用生成矩阵验证编码器正确性"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = _bit_reversal_indices(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(br):
        B[i, j] = 1
    G_full = (G @ B) % 2
    for trial in range(20):
        u = np.random.randint(0, 2, N)
        x_ref = (u @ G_full) % 2
        x_enc = polar_encode(u)
        if not np.array_equal(x_ref, x_enc):
            return False, u, x_ref, x_enc
    return True, None, None, None
