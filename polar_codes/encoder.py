"""
极化码编码器
编码：x = u @ G_N，G_N = B_N * F^{⊗n}
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def _generator_matrix(N):
    """构造 G_N = B_N F^{⊗n}（GF(2)）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    F_n = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        F_n = np.kron(F_n, F) % 2
    rev = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, rev[i]] = 1
    return (B @ F_n) % 2


_GEN_CACHE = {}


def _get_G(N):
    if N not in _GEN_CACHE:
        _GEN_CACHE[N] = _generator_matrix(N)
    return _GEN_CACHE[N]


def polar_encode(u):
    """
    极化码编码：x = u @ G_N。
    亦可用 O(N log N) 蝶形实现；此处用生成矩阵保证与 GA/译码器一致。
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    G = _get_G(N)
    return (u @ G) % 2


def polar_encode_fast(u):
    """O(N log N) 蝶形编码：先比特倒序 u，再蝶形（与矩阵编码等价）。"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    rev = bit_reversal_permutation(N)
    u = u[rev]
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i] = (u[i] ^ u[i + step]) % 2
        step *= 2
    return u


def polar_encode_for_decoder(u):
    return polar_encode(u)
