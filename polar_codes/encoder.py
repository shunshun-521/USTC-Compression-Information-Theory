"""
极化码编码器
与标准 Polar 编码（Sionna/Arikan 蝶形 XOR 调度）一致
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx & 1) << (n - 1))
    for b in range(1, n):
        rev |= ((idx >> b) & 1) << (n - 1 - b)
    return rev


def _gen_encode_indices(n):
    nb_stages = int(np.log2(n))
    ind_gather = np.ones([nb_stages, n + 1], dtype=np.int32) * n
    for s in range(nb_stages):
        ind_range = np.arange(n // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码。
    u: 长度 N 的源向量（冻结位为 0，信息位为 0/1）
    返回长度 N 的码字。
    """
    u = np.asarray(u, dtype=np.uint8)
    N = len(u)
    n = int(np.log2(N))
    ind_gather = _gen_encode_indices(N)
    x = np.concatenate([u, np.array([0], dtype=np.uint8)])
    for s in range(n):
        x_add = x[ind_gather[s, :]]
        x = np.bitwise_xor(x, x_add)
    return x[:N].astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u=", u, "x=", polar_encode(u))
