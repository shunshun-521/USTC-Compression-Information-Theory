"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for bit in range(n):
        rev |= ((indices >> bit) & 1) << (n - 1 - bit)
    return rev


def _butterfly_encode(u):
    """蝶形 XOR 编码（不含比特倒序）"""
    x = u.astype(np.int8, copy=True)
    n = int(np.log2(len(x)))
    for layer in range(n):
        step = 1 << layer
        for i in range(0, len(x), step << 1):
            x[i:i + step] ^= x[i + step:i + (step << 1)]
    return x


def polar_encode(u, apply_bit_reversal=False):
    """
    极化码编码（蝶形 XOR，默认与 SC 译码器配套）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）
        apply_bit_reversal: 是否在最后做比特倒序置换（G_N = B_N F^{\\otimes n}）

    返回：
        x: 长度为 N 的码字
    """
    x = _butterfly_encode(np.asarray(u, dtype=np.int8))
    if apply_bit_reversal:
        br = bit_reversal_permutation(len(x))
        x = x[br]
    return x.astype(int)


def polar_encode_with_bit_reversal(u):
    """G_N = B_N F^{\\otimes n} 形式的编码（含比特倒序）"""
    return polar_encode(u, apply_bit_reversal=True)


def build_generator_matrix(N, apply_bit_reversal=False):
    """构造生成矩阵 G_N = F^{\\otimes n}（或 B_N F^{\\otimes n}）"""
    n = int(np.log2(N))
    f = np.array([[1, 0], [1, 1]], dtype=int)
    g = f.copy()
    for _ in range(n - 1):
        g = np.kron(g, f)
    if apply_bit_reversal:
        br = bit_reversal_permutation(N)
        g = g[br, :]
    return g


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("polar_encode:", x)
    g = build_generator_matrix(4)
    x_mat = (u @ g) % 2
    print("matrix encode:", x_mat)
