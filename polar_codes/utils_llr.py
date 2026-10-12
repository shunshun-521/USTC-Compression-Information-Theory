"""LLR 顺序与编码器 bit-reversal 对齐"""
import numpy as np
from encoder import bit_reversal_permutation


def llr_to_decoder_order(llr_ch):
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=np.int64)
    inv[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]
