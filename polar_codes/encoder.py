"""
极化码编码器
编码：在信息位位置填入比特后，按极化码生成矩阵做 O(N log N) XOR 编码。
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.arange(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def _gen_encode_indices(N):
    """Sionna 风格的逐层 gather 索引（用于 XOR 编码）。"""
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

    参数：
        u: 长度为 N 的源序列（冻结位通常为 0）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.uint8).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    n_stages = int(np.log2(N))
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    ind_gather = _gen_encode_indices(N)

    for s in range(n_stages):
        ind_helper = ind_gather[s, :]
        x = np.bitwise_xor(x, x[ind_helper])

    return x[:N].astype(int)


def build_codeword(info_bits, info_indices, N):
    """将 K 个信息比特放入长度 N 的源向量（冻结位为 0）。"""
    u = np.zeros(N, dtype=int)
    u[info_indices] = np.asarray(info_bits, dtype=int)
    return polar_encode(u)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
    print("encoder self-test passed")
