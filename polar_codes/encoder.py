"""
极化码编码器
编码：x = u * F^{\\otimes n}，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)


def polar_encode(u):
    """
    极化码编码。
    x = u * F^{\\otimes n}（蝶形 XOR，无比特倒序）
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for s in range(n):
        step = 1 << s
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
    return u


def build_generator_matrix(N):
    """构建 G_N = F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=np.int8)
    result = np.array([[1]], dtype=np.int8)
    for _ in range(int(np.log2(N))):
        result = np.kron(result, F) % 2
    return result
