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
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def _gen_encode_indices(N):
    """Sionna 风格按阶段 XOR 索引。"""
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
    极化码编码（阶段 XOR，与 u @ G_N 一致）。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    n = int(np.log2(N))
    x = np.zeros(N + 1, dtype=np.int8)
    x[:N] = u
    ind_gather = _gen_encode_indices(N)
    for s in range(n):
        x[:N] ^= x[ind_gather[s, :N]]
    return x[:N]


def build_generator_matrix(N):
    """由编码器构造生成矩阵 G，满足 x = (u @ G) % 2。"""
    G = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        e = np.zeros(N, dtype=np.int8)
        e[i] = 1
        G[i] = polar_encode(e)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("polar_encode:", x)
    print("u @ G_N:", x_ref)
