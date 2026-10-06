"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组：out[i] = bit_reverse(i)"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=np.int64)
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        rev[i] = r
    return rev


def _butterfly_encode(u):
    """蝶形 XOR 编码（与 Arikan F^{⊗n} 一致，不含 B_N 置换）。"""
    u = np.asarray(u, dtype=np.int8).copy()
    N = len(u)
    n = int(np.log2(N))
    block = N
    for _ in range(n):
        half = block // 2
        for start in range(0, N, block):
            for k in range(half):
                idx = start + k
                u[idx] ^= u[idx + half]
        block = half
    return u


def polar_encode(u):
    """
    极化码编码：蝶形变换后施加比特倒序置换（x = u G_N, G_N = B_N F^{⊗n}）。
    """
    u_enc = _butterfly_encode(u)
    br = bit_reversal_permutation(len(u_enc))
    return u_enc[br].astype(int)


def polar_encode_no_br(u):
    """仅蝶形变换，用于 BP 早停重编码一致性检查。"""
    return _butterfly_encode(u).astype(int)
