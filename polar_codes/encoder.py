"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    rev = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int64)
    rev = rev[:, ::-1]
    weights = 1 << np.arange(n)
    return (rev * weights).sum(axis=1)


def _butterfly_encode_core(u):
    """Aff3ct 风格蝶形编码，等价于 c = G_N @ u (mod 2)。"""
    x = np.array(u, dtype=np.int8, copy=True)
    N = len(x)
    k = N // 2
    while k > 0:
        for j in range(0, N, 2 * k):
            for i in range(k):
                x[j + i] ^= x[k + j + i]
        k >>= 1
    return x


def polar_encode(u):
    """极化码编码（含比特倒序置换，用于信道发送顺序）。"""
    x = _butterfly_encode_core(u)
    br = bit_reversal_permutation(len(x))
    return x[br].astype(int)


def polar_encode_natural(u):
    """自然序码字（译码器内部一致性 / BP 早停）。"""
    return _butterfly_encode_core(u).astype(int)


def channel_to_decoder_llr(llr_ch):
    """将信道比特倒序后的 LLR 还原为译码器自然序。"""
    N = len(llr_ch)
    inv = np.argsort(bit_reversal_permutation(N))
    return llr_ch[inv]
