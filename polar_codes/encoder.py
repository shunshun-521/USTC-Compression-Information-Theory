"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=int)
    rev = ((idx[:, None] >> np.arange(n)) & 1).astype(int)
    rev = rev[:, ::-1]
    weights = 1 << np.arange(n)
    return (rev * weights).sum(axis=1)


def _butterfly_encode_inplace(u):
    """Arikan 蝶形：左半部分与右半部分异或（与 polarcodes 一致）。"""
    N = len(u)
    block = N
    while block > 1:
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half


def polar_encode(u):
    """
    极化码编码（Permuted-SC 约定：不在码字上做 B_N）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8).copy()
    _butterfly_encode_inplace(u)
    return u.astype(int)


def polar_encode_no_br(u):
    """与 polar_encode 相同（保留接口供 BP 早停等使用）。"""
    return polar_encode(u)
