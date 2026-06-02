"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{⊗ n}，与 Sionna PolarEncoder 一致的 XOR-gather 实现
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f'0{n}b')
        rev[i] = int(b[::-1], 2)
    return rev


def _gen_gather_indices(N):
    """预计算各层 XOR 编码的 gather 索引"""
    n = int(np.log2(N))
    ind_gather = np.ones((n, N + 1), dtype=np.int32) * N
    for s in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码（XOR-gather 蝶形，与标准 G_N 矩阵乘法等价）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int64)
    N = len(u)
    n = int(np.log2(N))
    x = np.zeros(N + 1, dtype=np.int64)
    x[:N] = u
    ind_gather = _gen_gather_indices(N)
    for s in range(n):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper].copy()
        x = np.bitwise_xor(x, x_add)
    return (x[:N] % 2).astype(int)


def polar_encode_matrix(u):
    """矩阵乘法编码 x = u @ G_N（用于校验，G_N = B_N F^{⊗n}）"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    br = bit_reversal_permutation(N)
    G = (np.eye(N, dtype=int)[br] @ G) % 2
    return (u @ G) % 2
