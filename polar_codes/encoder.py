"""
极化码编码器
编码：x = u * F^{\\otimes n}（mod 2），蝶形左 XOR 实现 O(N log N)
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(n):
        rev += ((indices >> i) & 1) << (n - 1 - i)
    return rev


def build_generator_matrix(N):
    """构造生成矩阵 F^{\\otimes n}（Arikan 标准核）。"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G


def polar_encode(u):
    """
    极化码编码：蝶形左 XOR，等价于 x = F^{\\otimes n} @ u (mod 2)。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = N
    for _ in range(n):
        step //= 2
        for p in range(0, N, 2 * step):
            for k in range(step):
                u[p + k] ^= u[p + k + step]
    return u


def polar_encode_verify(u):
    """矩阵乘法验证编码结果。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    if N not in _G_CACHE:
        _G_CACHE[N] = build_generator_matrix(N)
    return (_G_CACHE[N] @ u) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = polar_encode_verify(u)
    assert np.array_equal(x, expected), f"编码器错误: {x} vs {expected}"
    print("Encoder test passed:", x)
