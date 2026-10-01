"""
极化码编码器
编码：对 u 向量做极化变换（与蝶形 Kronecker 结构等价，按块大小递减实现）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = np.zeros(N, dtype=int)
    for i in idx:
        r = 0
        for bit in range(n):
            if (i >> bit) & 1:
                r |= 1 << (n - 1 - bit)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（非递归，与标准 polar_encode 块划分一致）。
    输入 u 为信息位+冻结位组成的源向量，输出编码后的发送比特（仍记为 u 域，译码器恢复同向量）。
    """
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    n = N
    while n > 1:
        n_split = n // 2
        for p in range(0, N, n):
            for k in range(n_split):
                l = p + k
                u[l] ^= u[l + n_split]
        n = n_split
    return u


def polar_encode_butterfly(u):
    """蝶形递增步长编码（用于 BP 早停重编码校验，与 polar_encode 等价）"""
    u = np.array(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    for stage in range(n):
        step = 1 << stage
        for i in range(0, N, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
    return u
