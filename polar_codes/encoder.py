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
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字

    实现：蝶形（butterfly）递归结构
        - 每层：相邻对 (u[i], u[i + step]) -> (u[i] XOR u[i+step], u[i+step])
        - 共 log2(N) 层
        - 最后做比特倒序置换（bit-reversal permutation）
    """
    u = np.array(u, dtype=int).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step *= 2
    br = bit_reversal_permutation(N)
    return u[br]


def polar_generator_matrix(N):
    """生成 G_N = B_N F^{\\otimes n}（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F) % 2
    br = bit_reversal_permutation(N)
    G = G[br, :]
    return G


def permute_llr_from_channel(llr):
    """将信道顺序的 LLR 置换为译码器内部顺序（比特倒序的逆）"""
    N = len(llr)
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[br] = np.arange(N)
    return llr[inv]
