"""Standalone adapter for mcba1n/polar-codes SCD (MIT license)."""
import numpy as np

from scd_vendor.decoder_utils import (
    active_bit_level,
    active_llr_level,
    hard_decision,
    lower_llr,
    upper_llr,
)
from scd_vendor.utils import bit_reversed


def vendor_sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen = set(int(i) for i in np.where(frozen_bits.astype(bool))[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for l in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

        if l in frozen:
            B[l, n] = 0
        else:
            B[l, n] = hard_decision(L[l, n])

        if l < N / 2:
            continue

        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(np.int8)
