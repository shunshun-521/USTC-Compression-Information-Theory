"""
极化码编码器
编码：x = u * G_N，与 Sionna PolarEncoder 的蝶形 XOR 阶段一致
"""
import numpy as np


def _gen_encode_indices(n):
    """Sionna 风格分阶段 gather 索引"""
    nb_stages = int(np.log2(n))
    ind_gather = np.ones((nb_stages, n + 1), dtype=np.int32) * n
    for s in range(nb_stages):
        ind_range = np.arange(n // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.arange(N, dtype=np.int64)
    for i in range(N):
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码：对长度 N 的源向量 u（含冻结位 0）做极化变换。
    """
    u = np.asarray(u, dtype=np.uint8)
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    ind_gather = _gen_encode_indices(N)
    nb_stages = int(np.log2(N))
    for s in range(nb_stages):
        ind_helper = ind_gather[s]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)
    return x[:N].astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    # 与 u @ (B_N F^{⊗n}) 一致时 x=[1,0,1,1]
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder test passed:", x)
