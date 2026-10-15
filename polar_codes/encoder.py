"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    step = 1
    while step < N:
        for i in range(0, N, 2 * step):
            for j in range(step):
                u[i + j] ^= u[i + j + step]
        step *= 2
    return u


def polar_encode(u):
    """
    极化码编码。

    采用 Arikan 蝶形结构（与 aff3ct / 标准 SC 因子图一致）。
    注：部分教材在输出端附加 B_N 置换；本实现与 SC/SCL/BP 译码器配套，
    等效于 G_N = F^{\\otimes n}（信道 LLR 按码字下标自然顺序）。
    """
    return _butterfly_encode(u).astype(int)


def polar_encode_with_bit_reversal(u):
    """蝶形编码后再做比特倒序置换（G = B_N F^{\\otimes n} 形式）"""
    br = bit_reversal_permutation(len(u))
    return _butterfly_encode(u)[br].astype(int)


def build_generator_matrix(N):
    """生成矩阵 G_N = F^{\\otimes n}（模 2）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    return G % 2
