"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def _gen_gather_indices(n):
    """Sionna 风格逐层 gather 索引。"""
    ind_gather = np.ones((int(np.log2(n)), n + 1), dtype=np.int32) * n
    for s in range(int(np.log2(n))):
        ind_range = np.arange(n // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码：u 为长度 N 的源向量（含冻结位 0），输出码字 x。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    x = np.zeros(N + 1, dtype=np.int8)
    x[:N] = u
    ind_gather = _gen_gather_indices(N)
    nb_stages = int(np.log2(N))
    for s in range(nb_stages):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper]
        x = (x ^ x_add).astype(np.int8)
    return x[:N]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder test passed:", x)
