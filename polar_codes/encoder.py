r"""
极化码编码器
编码：x = u * F^{\otimes n}（蝶形 XOR），与译码端约定一致
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组 rev，满足 out[rev[i]] = in[i]"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def _generator_matrix(N):
    """G_N = B_N F^{\\otimes n}，满足 x = (G_N @ u) % 2"""
    if N in _G_CACHE:
        return _G_CACHE[N]
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    Gf = F
    for _ in range(int(np.log2(N)) - 1):
        Gf = np.kron(Gf, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=np.int8)[br]
    G = (B @ Gf) % 2
    _G_CACHE[N] = G
    return G


def polar_encode(u):
    """
    极化码编码：蝶形 XOR，x = u F^{\\otimes n}（与 SC/SCL/BP 译码器一致）。
    等价于 x = B_N F^{\\otimes n} u 在比特倒序索引约定下的码字。
    """
    return polar_encode_butterfly(u)


def polar_encode_butterfly(u):
    """仅蝶形 F^{\\otimes n}（无 B_N），供 BP 早停重编码等内部使用。"""
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, step << 1):
            u[i:i + step] ^= u[i + step:i + (step << 1)]
    return u.astype(int)
