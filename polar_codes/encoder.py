"""
极化码编码器
编码：x = u * G_N，G_N = F^{⊗n}，F = [[1,0],[1,1]]
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([int(format(i, f'0{n}b')[::-1], 2) for i in range(N)])


def polar_generator_matrix(N):
    """构造 G_N = F^{⊗n}（GF(2)）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F) % 2
    return G


_G_CACHE = {}


def _get_G(N):
    if N not in _G_CACHE:
        _G_CACHE[N] = polar_generator_matrix(N)
    return _G_CACHE[N]


def polar_encode(u):
    """
    极化码编码：x = u @ G_N（GF(2)）。
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    G = _get_G(N)
    return (u @ G) % 2


if __name__ == '__main__':
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = (u @ polar_generator_matrix(4)) % 2
    print('u=', u, 'x=', x, 'G@u=', expected)
    assert np.array_equal(x, expected), f'编码器错误: {x} != {expected}'
    print('Encoder test passed.')
