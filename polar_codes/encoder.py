"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组。"""
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)


def _non_systematic_encode(message, n):
    """非系统化极化编码（与因子图一致的层序）。"""
    message = np.asarray(message, dtype=int).copy()
    for i in range(n - 1, -1, -1):
        step = 1 << (n - i - 1)
        groups = 1 << i
        for g in range(groups):
            start = 2 * g * step
            for p in range(step):
                message[p + start] ^= message[p + start + step]
    return message


def polar_encode(u):
    """
    极化码编码。

    对 u 做蝶形编码后施加比特倒序置换，得到 BPSK-AWGN 信道上的发送比特顺序。
    """
    u = np.asarray(u, dtype=int)
    N = len(u)
    n = int(np.log2(N))
    encoded = _non_systematic_encode(u, n)
    rev = bit_reversal_permutation(N)
    return encoded[rev]


def polar_encode_core(u):
    """仅蝶形编码（无比特倒序），供 BP 早停等内部一致性检查。"""
    u = np.asarray(u, dtype=int)
    n = int(np.log2(N := len(u)))
    return _non_systematic_encode(u, n)
