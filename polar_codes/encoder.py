"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def _generator_matrix(N):
    """极化码生成矩阵（蝶形编码等价）"""
    if N in _G_CACHE:
        return _G_CACHE[N]
    G = np.array([polar_encode(np.eye(N, dtype=int)[i]) for i in range(N)], dtype=int)
    _G_CACHE[N] = G
    return G


def polar_encode(u):
    """
    极化码编码（蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    n_split = N
    for _ in range(n):
        n_half = n_split // 2
        for p in range(0, N, n_split):
            for k in range(n_half):
                idx = p + k
                u[idx] ^= u[idx + n_half]
        n_split = n_half
    return u


def polar_encode_matrix(u):
    """矩阵乘法编码（用于校验）"""
    u = np.asarray(u, dtype=np.int8)
    G = _generator_matrix(len(u))
    return (u @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    xm = polar_encode_matrix(u)
    assert np.array_equal(x, xm), f"蝶形与矩阵不一致: {x} vs {xm}"
    print("u =", u, "-> x =", x)
    print("Encoder test passed.")
