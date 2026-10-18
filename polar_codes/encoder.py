"""
极化码编码器
编码：对 u 做 Arikan 蝶形变换（原地 XOR），与 u*G_N 等价（G_N = F^{\\otimes n}）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")
    rev = np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=np.int64)
    return rev


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字（与 u 经 Arikan 变换后相同）
    """
    x = np.asarray(u, dtype=np.int8).copy()
    N = len(x)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                x[idx] ^= x[idx + half]
        block = half
    return x


def generator_matrix(N):
    """G_N = F^{\\otimes n} (mod 2)。"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = generator_matrix(4)
    x_ref = (u.astype(np.int64) @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
