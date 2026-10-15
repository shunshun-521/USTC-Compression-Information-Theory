"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    indices = np.arange(N, dtype=int)
    rev = ((indices & 1) << (n - 1))
    for i in range(1, n):
        rev |= ((indices >> i) & 1) << (n - 1 - i)
    return rev


def polar_encode(u):
    """
    极化码编码（蝶形 XOR，与 SC 译码器因子图一致）。
    注：比特倒序置换 B_N 在译码端通过 LLR/比特索引调度吸收；
    `bit_reversal_permutation` 仍提供供报告与核对使用。
    """
    u = np.asarray(u, dtype=np.int_).copy()
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    block = N
    for _ in range(n):
        if block == 1:
            break
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] = (u[idx] + u[idx + half]) % 2
        block = half
    return u


def build_generator_matrix(N):
    """构造 G_N = F^{\\otimes n}（与 `polar_encode` 一致）"""
    n = int(np.log2(N))
    F = np.array([[1, 0], [1, 1]], dtype=np.int_)
    F_n = np.array([[1]], dtype=np.int_)
    for _ in range(n):
        F_n = np.kron(F_n, F)
    return F_n % 2
