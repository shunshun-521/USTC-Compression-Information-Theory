"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f'0{n}b')[::-1], 2)
    return rev


def polar_encode(u):
    """
    极化码编码。

    采用蝶形 XOR 结构（等价于 u 乘以 F 的 n 次 Kronecker 积），
    与 SC/SCL/BP 译码器的因子图约定一致。
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for start in range(0, N, block):
            for i in range(half):
                idx = start + i
                u[idx] ^= u[idx + half]
        block = half
    return u


if __name__ == "__main__":
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    while G.shape[0] < 4:
        G = np.kron(G, F)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = (u @ G) % 2
    print("u =", u, "-> x =", x, "(ref G:", x_ref, ")")
    assert np.array_equal(x, x_ref), f"编码器错误: {x}"
