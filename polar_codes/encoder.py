"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗ n}
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _generator_matrix(N):
    if N not in _G_CACHE:
        F = np.array([[1, 0], [1, 1]], dtype=np.int8)
        G = np.array([[1]], dtype=np.int8)
        while G.shape[0] < N:
            G = np.kron(G, F)
        _G_CACHE[N] = G.astype(np.int8)
    return _G_CACHE[N]


def polar_encode(u):
    """
    极化码编码：x = u * G_N（GF(2)），与 SC/SCL/BP 因子图译码器一致。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = _generator_matrix(N)
    return (u @ G) % 2


def polar_encode_bit_reversed(u):
    """x = u * B_N * G_N（含比特倒序置换的等价形式）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    print("u =", u, "-> x =", polar_encode(u))
