"""
极化码编码器
编码：x = u * G_N，利用蝶形 / 分阶段 XOR 结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def _precompute_encode_indices(N):
    """分阶段 XOR 编码索引（与 5G/Sionna 极化编码等价）。"""
    n = int(np.log2(N))
    ind_gather = np.ones((n, N + 1), dtype=np.int32) * N
    for s in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


_ENCODE_CACHE = {}


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位应为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.uint8)
    N = len(u)
    if N not in _ENCODE_CACHE:
        _ENCODE_CACHE[N] = _precompute_encode_indices(N)
    ind_gather = _ENCODE_CACHE[N]

    x = np.concatenate([u, np.array([0], dtype=np.uint8)])
    for s in range(int(np.log2(N))):
        x = np.bitwise_xor(x, x[ind_gather[s]])
    return x[:N].astype(int)


def polar_encode_butterfly_bitrev(u):
    """蝶形 + 比特倒序编码（与 polar_encode 在标准 G_N 下等价，供对照）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            u[i + step : i + 2 * step] ^= u[i : i + step]
        step <<= 1
    rev = bit_reversal_permutation(N)
    return u[rev].astype(int)


def polar_generator_matrix(N):
    """GF(2) 生成矩阵 G = B_N F^{\\otimes n}，用于校验。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=np.int8)
    for i, r in enumerate(rev):
        B[i, r] = 1
    return (B @ G) % 2
