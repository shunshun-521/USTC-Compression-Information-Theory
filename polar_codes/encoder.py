"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _gen_gather_indices(N):
    """Sionna 风格的蝶形 XOR 索引"""
    nb_stages = int(np.log2(N))
    ind_gather = np.ones((nb_stages, N + 1), dtype=np.int32) * N
    for s in range(nb_stages):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2 ** s)
        ind_origin = ind_dest + 2 ** s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码（Sionna 风格蝶形 XOR，无额外比特倒序）。
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = len(u)
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    ind_gather = _gen_gather_indices(N)
    for s in range(int(np.log2(N))):
        ind_helper = ind_gather[s, :]
        x = np.bitwise_xor(x, x[ind_helper])
    return x[:N].astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
