"""
极化码编码器：蝶形极化变换（与标准 SC 因子图一致）
"""
import numpy as np

_G_CACHE = {}


def _bit_reverse_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([_bit_reverse_index(i, n) for i in range(N)], dtype=np.int64)


def _build_f_tensor(N):
    """F^{⊗n}，用于校验/报告"""
    if N in _G_CACHE:
        return _G_CACHE[N]
    n = int(np.log2(N))
    if n == 1:
        G = np.array([[1, 0], [1, 1]], dtype=np.int8)
    else:
        Fp = _build_f_tensor(N // 2)
        h = N // 2
        G = np.zeros((N, N), dtype=np.int8)
        G[:h, :h] = Fp
        G[h:, :h] = Fp
        G[h:, h:] = Fp
    _G_CACHE[N] = G
    return G


def polar_encode(u):
    """
    极化编码：对 u 执行 log2(N) 层蝶形 XOR（O(N log N)）。
    输出即为信道发送的码字比特（与 mcba1n/polar-codes 一致）。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n_block = N
    for _ in range(int(np.log2(N))):
        if n_block == 1:
            break
        n_split = n_block // 2
        for p in range(0, N, n_block):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n_block = n_split
    return u.astype(np.int8)


def polar_encode_matrix(u):
    """u @ F^{⊗n} (mod 2)，与蝶形编码等价"""
    u = np.asarray(u, dtype=np.int8)
    G = _build_f_tensor(len(u))
    return (u @ G) % 2
