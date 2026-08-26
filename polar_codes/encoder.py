"""
极化码编码器
编码：蝶形 XOR（分块左加右），与置换 SC 译码配套
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(i, n):
    """对标量索引 i 做 n 位比特倒序"""
    return int(f"{i:0{n}b}"[::-1], 2)


def polar_encode(u):
    """
    极化码编码（O(N log N) 蝶形结构）。
    对每个阶段将右半分区 XOR 到左半分区。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def _generator_matrix(N):
    """生成矩阵（用于 BP 早停重编码验证）。"""
    if N in _G_CACHE:
        return _G_CACHE[N]
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    _G_CACHE[N] = F_n.astype(np.int8)
    return _G_CACHE[N]


def polar_encode_matrix(u):
    """矩阵编码 x = u @ F^{⊗n}（与蝶形编码等价）。"""
    u = np.asarray(u, dtype=int)
    G = _generator_matrix(len(u))
    return (u.astype(np.int64) @ G.astype(np.int64)) % 2
