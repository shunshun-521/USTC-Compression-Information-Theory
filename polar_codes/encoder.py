"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np

_G_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        b = format(i, f"0{n}b")
        rev[i] = int(b[::-1], 2)
    return rev


def _get_generator_matrix(N):
    """Arikan 生成矩阵 G_N = F^{\\otimes n}, F=[[1,1],[0,1]]"""
    if N in _G_CACHE:
        return _G_CACHE[N]
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F)
    _G_CACHE[N] = G
    return G


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 Arikan F^{\\otimes n} 一致）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    return polar_encode_butterfly(u)


def polar_encode_butterfly(u):
    """蝶形结构 O(N log N) 编码"""
    u = np.asarray(u, dtype=np.int8).copy().ravel()
    N = len(u)
    n = int(np.log2(N))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u.astype(int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = (u @ _get_generator_matrix(4)) % 2
    print("encode test:", x)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
