"""
极化码编码器
编码：x = u * G_N = u * F^{\\otimes n}，蝶形结构 O(N log N)
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def bit_reversed(x, n):
    """对标量索引 x 做比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码：u[l] ^= u[l + step]（蝶形 XOR），与标准 F^{\\otimes n} 生成矩阵一致。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                l = start + k
                u[l] ^= u[l + half]
        block = half
    return u


def build_generator_matrix(N):
    """构造 Arikan 生成矩阵 F^{\\otimes n}"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print(f"u={u} -> x={x}, matrix={x_mat}")
    assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
    print("Encoder test passed.")
