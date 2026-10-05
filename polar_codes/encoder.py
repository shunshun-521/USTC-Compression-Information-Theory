"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in idx:
        r = 0
        x = i
        for _ in range(n):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码（蝶形结构 O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）
        apply_bit_reversal: 是否在末尾做 B_N 比特倒序置换（默认 False，
            与 SC/SCL/BP 译码器内部索引一致；设为 True 可得到 x = u B_N F^{⊗n}）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            u[i:i + step] ^= u[i + step:i + 2 * step]
        step <<= 1
    if apply_bit_reversal:
        br = bit_reversal_permutation(N)
        u = u[br]
    return u


def polar_encode_matrix(N):
    """生成 N×N 生成矩阵（用于验证）"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    for _ in range(int(np.log2(N)) - 1):
        G = np.kron(G, F)
    B = np.zeros((N, N), dtype=int)
    br = bit_reversal_permutation(N)
    for i, j in enumerate(br):
        B[i, j] = 1
    return (B @ G) % 2


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode", u, "->", x)
    G = polar_encode_matrix(4)
    x2 = (u @ G) % 2
    print("matrix", x2)
