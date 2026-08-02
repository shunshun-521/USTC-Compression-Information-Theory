"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _gen_gather_indices(n):
    """生成与 Sionna PolarEncoder 一致的 XOR 索引"""
    ind_gather = np.ones((n, (1 << n) + 1), dtype=np.int32) * (1 << n)
    N = 1 << n
    for s in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码（Sionna 兼容的 XOR-gather 结构）。
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    # 末尾填充一位，与 Sionna 实现一致
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u.astype(np.uint8)

    ind_gather = _gen_gather_indices(n)
    for s in range(n):
        x_add = x[ind_gather[s, :]]
        x = np.bitwise_xor(x, x_add)

    return x[:N].astype(int)


def build_generator_matrix(N):
    """构建生成矩阵（用于验证）"""
    n = int(np.log2(N))
    u = np.eye(N, dtype=int)
    return np.array([polar_encode(u[i]) for i in range(N)], dtype=int)
