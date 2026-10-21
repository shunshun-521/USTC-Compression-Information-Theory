"""
极化码编码器：x = u F^{⊗n}（Arikan 极化变换，O(N log N)）
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """比特倒序置换索引"""
    n = int(np.log2(N))
    indices = np.empty(N, dtype=int)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        indices[i] = r
    return indices


def _polar_transform_recursive(u):
    n = len(u)
    if n == 1:
        return list(u)
    half = n // 2
    upper = [int(u[i]) ^ int(u[i + half]) for i in range(half)]
    return _polar_transform_recursive(upper) + _polar_transform_recursive(
        list(u[half:])
    )


def polar_encode(u):
    """
    极化码编码。
    与生成矩阵 G = F^{⊗n}（无额外比特倒序）一致。
    """
    u = np.asarray(u, dtype=int)
    return np.array(_polar_transform_recursive(u), dtype=int)


def _generator_matrix(N):
    if N in _G_CACHE:
        return _G_CACHE[N]
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = np.array([[1]], dtype=np.int8)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    _G_CACHE[N] = G % 2
    return _G_CACHE[N]


def polar_encode_matrix(u):
    """矩阵乘法编码（校验用）"""
    u = np.asarray(u, dtype=np.int8)
    return (u @ _generator_matrix(len(u))) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"编码不一致: {x} vs {xm}"
    print("u=", u, "x=", x)
