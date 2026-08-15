"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码蝶形编码，复杂度 O(N log N)。

    输出 x = u * F^{⊗n}；比特倒序矩阵 B_N 的作用由
    SC/SCL 译码器在比特倒序索引顺序下完成，整体等价于 x = u * B_N * F^{⊗n}。
    """
    return polar_encode_butterfly(u)


def polar_encode_butterfly(u):
    """蝶形 XOR 编码。"""
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    if N & (N - 1):
        raise ValueError("N must be a power of 2")

    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode_matrix(u):
    """使用生成矩阵 B_N * F^{⊗n} 编码（用于验证）。"""
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    rev = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[rev]
    GN = (B @ G) % 2
    return (u @ GN) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u)
    print("x (butterfly) =", x)
    # 蝶形编码等价于 u @ F^{⊗n}
    N = len(u)
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(n - 1):
        G = np.kron(G, F)
    assert np.array_equal(x, u @ G % 2), f"编码器错误: {x}"
    print("编码器校验通过")
