"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)],
        dtype=int,
    )


def polar_encode(u):
    """
    极化码编码（蝶形结构，与 SC 译码器配套的自然序码字）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    注：SC 译码器按比特倒序逐位译码，与此处自然序编码配套。
    若需显式比特倒序输出，可调用 polar_encode_with_br(u)。
    """
    u = np.asarray(u, dtype=int).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            left = u[i:i + step]
            right = u[i + step:i + 2 * step]
            u[i:i + step] = (left ^ right) % 2
        step *= 2
    return u


def polar_encode_with_br(u):
    """蝶形编码后再做比特倒序置换（显式 B_N 映射）。"""
    x = polar_encode(u)
    return x[bit_reversal_permutation(len(x))]


def polar_generator_matrix(N):
    """返回 N×N 生成矩阵 F^{⊗n}（自然序，用于验证）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.array([[1]], dtype=int)
    n = int(np.log2(N))
    for _ in range(n):
        G = np.kron(G, F)
    return G
