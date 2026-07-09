"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    if 2 ** n != N:
        raise ValueError(f"N={N} must be a power of 2")
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _butterfly_encode(u):
    """蝶形 XOR 编码（块递归，与信道端极化因子图一致）"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    block = N
    for _ in range(int(np.log2(N))):
        if block == 1:
            break
        half = block // 2
        for base in range(0, N, block):
            for k in range(half):
                idx = base + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode(u):
    """
    极化码编码（含比特倒序置换）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = _butterfly_encode(u)
    br = bit_reversal_permutation(len(u))
    return u[br].astype(int)


def polar_encode_core(u):
    """蝶形编码结果（不做比特倒序），供译码器 LLR 对齐使用"""
    return _butterfly_encode(u).astype(int)
