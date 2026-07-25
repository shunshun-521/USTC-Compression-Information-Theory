r"""
极化码编码器
编码：x = u * G_N，G_N = F^{\otimes n}，利用块 XOR 结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int("".join(reversed(format(i, f"0{n}b"))), 2)
    return rev


def polar_encode(u):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字，满足 x = u @ G_N (mod 2)，G_N = F^{\otimes n}
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    if 2**n != N:
        raise ValueError("N must be a power of 2")

    v = u.copy()
    for layer in range(1, n + 1):
        block = 1 << layer
        half = block >> 1
        for block_idx in range(N // block):
            base = block_idx * block
            for pos in range(half):
                v[base + pos] ^= v[base + half + pos]
    return v


def build_generator_matrix(N):
    r"""构造 G_N = F^{\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    print("u =", u)
    print("x =", x)
    print("u@G =", (u @ G) % 2)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"
    print("Encoder test passed.")
