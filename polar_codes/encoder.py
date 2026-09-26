"""
极化码编码器
采用与标准极化码因子图一致的逐层 XOR（与 5G/Sionna 编码结构相同）
"""
import numpy as np


def _gen_encode_indices(n):
    nb_stages = int(np.log2(n))
    ind_gather = np.ones([nb_stages, n + 1], dtype=np.int32) * n
    for s in range(nb_stages):
        ind_range = np.arange(n // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


_ENCODE_CACHE = {}


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
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


def polar_encode(u):
    """
    极化码编码：u 为长度 N 的源向量（冻结位 + 信息位），输出长度 N 的码字。
    """
    u = np.asarray(u, dtype=np.uint8)
    n = len(u)
    if n & (n - 1):
        raise ValueError("N must be a power of 2")
    if n not in _ENCODE_CACHE:
        _ENCODE_CACHE[n] = _gen_encode_indices(n)
    ind_gather = _ENCODE_CACHE[n]
    x = np.zeros(n + 1, dtype=np.uint8)
    x[:n] = u
    for s in range(int(np.log2(n))):
        helper = ind_gather[s, :]
        x = np.bitwise_xor(x, x[helper])
    return x[:n].astype(np.int8)


def align_llr_to_decoder(llr_ch):
    """信道 LLR 与码字比特一一对应，无需额外置换"""
    return np.asarray(llr_ch, dtype=np.float64)
