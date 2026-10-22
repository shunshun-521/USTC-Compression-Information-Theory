"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    return np.array([int(bin(i)[2:].zfill(n)[::-1], 2) for i in range(N)], dtype=int)


def _gen_encode_indices(n):
    """Sionna 风格按阶段 XOR 索引（与极化生成矩阵一致）"""
    ind_gather = np.ones((n, 2**n + 1), dtype=np.int32) * (2**n)
    N = 2**n
    for s in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码：将长度 N 的源向量 u（冻结位为 0）映射为码字 x。
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = u.size
    if N & (N - 1):
        raise ValueError("N must be a power of 2")
    n = int(np.log2(N))
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    ind_gather = _gen_encode_indices(n)
    for s in range(n):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)
    return x[:N].astype(int)


def polar_generator_matrix(N):
    """生成矩阵 G_N（用于校验）"""
    n = int(np.log2(N))
    u = np.eye(N, dtype=int)
    return np.stack([polar_encode(u[i]) for i in range(N)], axis=0) % 2
