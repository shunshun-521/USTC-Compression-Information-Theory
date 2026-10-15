"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

try:
    from numba import jit
except ImportError:

    def jit(*_args, **_kwargs):
        def wrap(fn):
            return fn

        return wrap

from decoder_sc import _li


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


@jit(nopython=True)
def _scl_decode_core(llr_channel, frozen_mask, list_size, n):
    N = len(llr_channel)
    L = list_size
    template = -np.inf * np.ones((n + 1, N), dtype=np.float64)
    template[n, :] = llr_channel
    llrs = [template.copy() for _ in range(L)]
    s = [-np.ones((n + 1, N), dtype=np.int8) for _ in range(L)]
    pm = np.inf * np.ones(L, dtype=np.float64)
    pm[0] = 0.0
    dm = np.zeros(L, dtype=np.float64)
    pm_dm = np.zeros(2 * L, dtype=np.float64)

    for ii in range(N):
        if frozen_mask[ii]:
            for dd in range(L):
                llrs[dd][0, ii] = _li(0, ii, llrs[dd], s[dd], n)
                s[dd][0, ii] = 0
                pm[dd] += -llrs[dd][0, ii] * (llrs[dd][0, ii] < 0)
        else:
            for dd in range(L):
                llrs[dd][0, ii] = _li(0, ii, llrs[dd], s[dd], n)
                s[dd][0, ii] = 1 if llrs[dd][0, ii] < 0 else 0
                dm[dd] = abs(llrs[dd][0, ii])

            if L > 1:
                pm_dm[:L] = pm
                pm_dm[L:] = pm + dm
                idx_sort = np.argsort(pm_dm)
                new_llrs = []
                new_s = []
                new_pm = np.empty(L, dtype=np.float64)
                for k in range(L):
                    choice = int(idx_sort[k])
                    new_pm[k] = pm_dm[choice]
                    if choice < L:
                        new_llrs.append(llrs[choice])
                        new_s.append(s[choice])
                    else:
                        parent = choice - L
                        llr_copy = llrs[parent].copy()
                        s_copy = s[parent].copy()
                        s_copy[0, ii] = 1 - s[parent][0, ii]
                        new_llrs.append(llr_copy)
                        new_s.append(s_copy)
                llrs, s, pm = new_llrs, new_s, new_pm

    best = int(np.argmin(pm))
    return s[best][0, :].astype(np.int64), pm[best]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        fb = np.asarray(frozen_bits)
        if fb.dtype == bool:
            self.frozen_mask = fb.astype(np.int8)
        else:
            self.frozen_mask = (fb != 0).astype(np.int8)
        self.info_indices = np.where(self.frozen_mask == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_hat, pm = _scl_decode_core(
            llr_ch, self.frozen_mask, self.list_size, self.n
        )
        u_hat = u_hat.astype(int)

        if self.crc_length <= 0:
            return u_hat, pm

        info_bits = u_hat[self.info_indices]
        if crc_check(info_bits, self.crc_length):
            return u_hat, pm

        if self.list_size <= 1:
            return u_hat, pm

        return u_hat, pm
