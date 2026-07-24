r"""
极化码编码器
编码：x = u * G_N，利用蝶形 XOR 结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)])


def _gen_encode_indices(N):
    """预计算编码阶段的 gather 索引（与极化码标准实现一致）"""
    n = int(np.log2(N))
    ind_gather = np.ones((n, N + 1), dtype=np.int32) * N
    for stage in range(n):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** stage)
        ind_origin = ind_dest + 2 ** stage
        ind_gather[stage, ind_dest] = ind_origin
    return ind_gather


_ENCODE_CACHE = {}


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    if N not in _ENCODE_CACHE:
        _ENCODE_CACHE[N] = _gen_encode_indices(N)
    ind_gather = _ENCODE_CACHE[N]

    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u.astype(np.uint8)
    for stage in range(ind_gather.shape[0]):
        x = np.bitwise_xor(x, x[ind_gather[stage]])
    return x[:N].astype(int)


def build_generator_matrix(N):
    """构造生成矩阵（用于校验）"""
    u = np.eye(N, dtype=int)
    return np.array([polar_encode(u[i]) for i in range(N)], dtype=int)
