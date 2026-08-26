"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def inverse_bit_reversal_permutation(N):
    """返回 bit-reversal 的逆置换索引"""
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[br] = np.arange(N)
    return inv


def polar_transform_core(u):
    """Kronecker F 蝶形编码（不含比特倒序）"""
    x = np.array(u, dtype=np.int8, copy=True)
    length = x.size
    step = 1
    while step < length:
        for start in range(0, length, 2 * step):
            x[start : start + step] ^= x[start + step : start + 2 * step]
        step <<= 1
    return x


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。
    x = B_N * (u @ F^{\otimes n})
    """
    x = polar_transform_core(u)
    return x[bit_reversal_permutation(len(x))]


def build_generator_matrix(N):
    """构造 G_N = B_N F^{\otimes n}（GF(2)），用于验证"""
    n = int(np.log2(N))
    f = np.array([[1, 0], [1, 1]], dtype=np.int8)
    g = f.copy()
    for _ in range(n - 1):
        g = np.kron(g, f)
    br = bit_reversal_permutation(N)
    return g[br, :]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    g = build_generator_matrix(4)
    x_mat = (u @ g) % 2
    print("u =", u)
    print("polar_encode =", x)
    print("matrix encode =", x_mat)
