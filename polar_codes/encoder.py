"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def bit_reversed(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（Arikan 蝶形结构，输出为信道发送顺序）。

    实现与标准极化码一致：u 经 F^{\\otimes n} 蝶形变换后即为码字 x。
    比特倒序置换体现在译码器的索引映射中（见 decoder_sc）。
    """
    x = np.array(u, dtype=np.int8, copy=True)
    n = int(np.log2(len(x)))
    block = len(x)
    while block > 1:
        half = block // 2
        for base in range(0, len(x), block):
            for k in range(half):
                idx = base + k
                x[idx] ^= x[idx + half]
        block = half
    return x


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_mat = (u @ G) % 2
    print("butterfly encode:", x)
    print("matrix encode:", x_mat)
