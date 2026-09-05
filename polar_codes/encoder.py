"""
极化码编码器
编码：利用 Arikan 蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        rev[i] = int(f"{int(i):0{n}b}"[::-1], 2)
    return rev


def bit_reversed_index(x, n):
    """对标量索引 x 做 n 位比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def polar_encode(u):
    """
    极化码编码（非系统化，Arikan 蝶形 XOR）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    stage_len = N
    while stage_len > 1:
        half = stage_len // 2
        for block_start in range(0, N, stage_len):
            for k in range(half):
                idx = block_start + k
                u[idx] ^= u[idx + half]
        stage_len = half
    return u


def polar_generator_matrix(N):
    """返回 GF(2) 生成矩阵 F^{\\otimes n}，F=[[1,1],[0,1]]。"""
    F = np.array([[1, 1], [0, 1]], dtype=np.int8)
    G = F.copy()
    while G.shape[0] < N:
        G = np.kron(F, G)
    return G.astype(np.int8) % 2


def polar_encode_reference(u):
    """矩阵乘法编码，用于单元测试。"""
    u = np.asarray(u, dtype=np.int8)
    G = polar_generator_matrix(len(u))
    return (G.astype(np.int64) @ u.astype(np.int64)) % 2


if __name__ == "__main__":
    u = np.array([0, 1, 0, 0])
    x = polar_encode(u)
    xref = polar_encode_reference(u)
    print("u =", u)
    print("x =", x)
    assert np.array_equal(x, xref), f"编码器错误: {x} vs {xref}"
    assert np.array_equal(x, [1, 1, 0, 0]), f"编码器错误: {x}"
    print("Encoder test passed.")
