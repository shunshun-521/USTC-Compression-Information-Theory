"""
极化码编码器
编码：x = u * G_N，G_N = F^{\otimes n}
"""
import numpy as np

_G_CACHE = {}


def _generator_matrix(N):
    """F^{\otimes n} 生成矩阵。"""
    if N not in _G_CACHE:
        F = np.array([[1, 0], [1, 1]], dtype=int)
        G = np.array([[1]], dtype=int)
        while G.shape[0] < N:
            G = np.kron(G, F)
        _G_CACHE[N] = G % 2
    return _G_CACHE[N]


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """极化码编码：x = u @ G_N (mod 2)。"""
    return polar_encode_matrix(u)


def polar_encode_matrix(u):
    """矩阵形式 x = u @ G_N。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    return (u @ _generator_matrix(N)) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder test passed, x=", x)
