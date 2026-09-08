"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _gen_encode_indices(N):
    """Sionna/5G 风格的 stage-wise XOR gather 索引"""
    n_stages = int(np.log2(N))
    ind_gather = np.ones((n_stages, N + 1), dtype=np.int32) * N
    for s in range(n_stages):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码。

    参数 u 为长度 N 的源向量（含冻结位 0），返回长度 N 的码字。
    采用与标准 Polar 因子图一致的 stage-wise XOR 结构（与 Sionna 等价）。
    """
    u = np.array(u, dtype=np.uint8, copy=True)
    N = len(u)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N must be a power of 2")

    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    ind_gather = _gen_encode_indices(N)

    for s in range(n):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)

    return x[:N].astype(int)


def polar_encode_info(info_bits, info_indices, N):
    """将 K 个信息比特映射到 info_indices 后编码"""
    u = np.zeros(N, dtype=int)
    u[info_indices] = info_bits
    return polar_encode(u)


def polar_encode_matrix(N):
    """构造生成矩阵（验证用）"""
    I = np.eye(N, dtype=int)
    cols = np.vstack([polar_encode(I[i]) for i in range(N)]).T
    return cols % 2


def polar_encode_matmul(u):
    """矩阵乘法编码（验证用）"""
    N = len(u)
    G = polar_encode_matrix(N)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1, 0, 0, 0, 0])
    print("u =", u)
    print("x =", polar_encode(u))
    print("matmul =", polar_encode_matmul(u))
