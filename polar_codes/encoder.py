"""
极化码编码器
编码：x = u * G_N，G_N = B_N F^{\\otimes n}
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def _generator_matrix(N):
    if N in _G_CACHE:
        return _G_CACHE[N]
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    brp = bit_reversal_permutation(N)
    for i in range(N):
        B[i, brp[i]] = 1
    Gn = (B @ G) % 2
    _G_CACHE[N] = Gn
    return Gn


def polar_encode(u):
    """
    极化码编码。
    u: 长度为 N 的源向量（信息位 + 冻结位，冻结位应为 0）
    返回长度为 N 的码字 x = u G_N (mod 2)
    """
    u = np.asarray(u, dtype=int).ravel()
    N = len(u)
    Gn = _generator_matrix(N)
    return (u @ Gn) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u=", u, "x=", x)
