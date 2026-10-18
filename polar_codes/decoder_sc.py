"""
极化码 SC（串行抵消）译码器
信道 LLR 经比特倒序置换后，在自然索引顺序下做 SC 译码。
"""
import math
import numpy as np


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, Lb - La)


def channel_llr_to_decode_domain(llr_ch):
    """将信道 LLR 映射到 SC 译码域（与编码端 B_N 配套）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    out = np.zeros(N, dtype=np.float64)
    for i in range(N):
        out[bit_reversed(i, n)] = llr_ch[i]
    return out


def precompute_sc_indices(N):
    """保留接口：返回译码域置换说明。"""
    n = int(math.log2(N))
    return [bit_reversed(i, n) for i in range(N)], None, None


def _sc_recursive(llr, frozen_bits, indices):
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    if len(indices) == 1:
        i = indices[0]
        bit = 0 if frozen_bits[i] else (0 if llr[0] >= 0 else 1)
        return [bit], {i: bit}

    half = len(indices) // 2
    left_llr = f_operation(llr[:half], llr[half:])
    enc_left, dec_left = _sc_recursive(left_llr, frozen_bits, indices[:half])
    left_bits = np.asarray(enc_left, dtype=int)
    right_llr = g_operation(llr[:half], llr[half:], left_bits)
    enc_right, dec_right = _sc_recursive(right_llr, frozen_bits, indices[half:])
    dec_left.update(dec_right)
    combined = [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right
    return combined, dec_left


def _decode_natural(llr, frozen_bits):
    _, decisions = _sc_recursive(np.asarray(llr, dtype=np.float64), frozen_bits, list(range(len(llr))))
    u_hat = np.zeros(len(llr), dtype=int)
    for i, v in decisions.items():
        u_hat[i] = v
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    llr_dec = channel_llr_to_decode_domain(llr)
    return _decode_natural(llr_dec, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归入口：当前使用高效递归内核（译码域内）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
