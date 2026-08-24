"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
G_N = F^{\\otimes n}，F = [[1,1],[0,1]]（Arikan 标准核）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def polar_encode(u):
    """
    极化码编码（蝶形结构，复杂度 O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=int)
    x = u.copy()
    N = len(x)
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                x[p + k] ^= x[p + k + half]
        block = half
    return x


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}"""
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    if N > 1:
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    print("u:", u)
    print("polar_encode:", x)
    print("matrix ref (u @ G):", x_ref)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
