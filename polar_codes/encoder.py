"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
（与 F^{\\otimes n} 生成矩阵一致，不在码字上做额外比特倒序）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码（蝶形 XOR 结构，复杂度 O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(1, n + 1):
        block = 1 << stage
        half = block // 2
        for base in range(0, N, block):
            u[base : base + half] ^= u[base + half : base + block]
    return (u & 1).astype(int)


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（用于校验）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    for _ in range(int(np.log2(N))):
        G = np.kron(G, F)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", u, "->", x)
    G = build_generator_matrix(4)
    print("matrix:", (u @ G) % 2)
