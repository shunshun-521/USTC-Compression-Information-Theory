"""
极化码编码器
编码：x = u * G_N，G_N = F 的 n 次 Kronecker 积
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _generator_matrix(N):
    if N not in _G_CACHE:
        n = int(np.log2(N))
        F = np.array([[1, 0], [1, 1]], dtype=np.int8)
        G = F.copy()
        for _ in range(n - 1):
            G = np.kron(G, F)
        _G_CACHE[N] = G
    return _G_CACHE[N]


def polar_encode(u):
    """
    极化码编码。
    x = u * G_N，G_N = F^{\otimes n}
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    G = _generator_matrix(N)
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)
