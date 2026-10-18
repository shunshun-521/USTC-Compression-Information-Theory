"""
极化码编码器
编码：u（含冻结位）经多级 XOR 得到码字（与标准 PolarEncoder 一致）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        for b in range(n):
            if (i >> b) & 1:
                r |= 1 << (n - 1 - b)
        rev[i] = r
    return rev


def _gen_encode_indices(N):
    """Sionna 风格的逐级 XOR 索引（N 为码长）。"""
    nb_stages = int(np.log2(N))
    ind_gather = np.ones((nb_stages, N + 1), dtype=np.int32) * N
    for s in range(nb_stages):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位，冻结位为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u.astype(np.uint8)
    ind_gather = _gen_encode_indices(N)
    for s in range(n):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)
    return x[:N].astype(np.int8)


def build_generator_matrix(N):
    """构建 G_N，用于单元测试。"""
    G = np.zeros((N, N), dtype=np.int8)
    for j in range(N):
        u = np.zeros(N, dtype=np.int8)
        u[j] = 1
        G[j] = polar_encode(u)
    return G
