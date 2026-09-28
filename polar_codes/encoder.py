"""
极化码编码器
编码：scatter 信息位 + 分阶段 XOR（与标准 Polar 编码等价）
"""
import numpy as np

_GEN_INDICES = {}


def _gen_indices(n):
    if n in _GEN_INDICES:
        return _GEN_INDICES[n]
    nb_stages = int(np.log2(n))
    ind_gather = np.ones((nb_stages, n + 1), dtype=np.int32) * n
    for s in range(nb_stages):
        ind_range = np.arange(n // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    _GEN_INDICES[n] = ind_gather
    return ind_gather


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


def polar_encode(u, info_indices=None):
    """
    极化码编码。
    u: 长度 N 的源向量（冻结位为 0），或提供 info_indices 时仅信息位已填入。
    """
    u = np.asarray(u, dtype=np.int64).copy()
    N = len(u)
    n = int(np.log2(N))
    x = np.zeros(N + 1, dtype=np.int64)
    x[:N] = u % 2
    ind_gather = _gen_indices(N)
    for s in range(n):
        ind_helper = ind_gather[s]
        x_add = x[ind_helper]
        x[: N + 1] = np.bitwise_xor(x, x_add)
    return x[:N] % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("encode test:", polar_encode(u))
