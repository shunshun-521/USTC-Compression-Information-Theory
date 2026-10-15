"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import _f_node_minsum, _g_node, _li, _s_updater, _b_check


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_polynomial(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L = self.list_size

        llrs_list = []
        s_list = []
        for _ in range(L):
            llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
            llrs[n, :] = llr_ch
            llrs_list.append(llrs)
            s_list.append(np.full((n + 1, N), -1, dtype=np.int8))

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0
        dm = np.zeros(L, dtype=np.float64)

        for bit_idx in range(N):
            if self.frozen_bits[bit_idx]:
                for d in range(L):
                    llr0 = _li(0, bit_idx, llrs_list[d], s_list[d])
                    s_list[d][0, bit_idx] = 0
                    pm[d] += -llr0 * (llr0 < 0)
            else:
                for d in range(L):
                    llr0 = _li(0, bit_idx, llrs_list[d], s_list[d])
                    s_list[d][0, bit_idx] = 1 if llr0 < 0 else 0
                    dm[d] = abs(llr0)

                if L > 1:
                    pm_dm = np.concatenate([pm, pm + dm])
                    idx_sort = np.argsort(pm_dm)
                    idx_keep = idx_sort[:L]
                    new_llrs = []
                    new_s = []
                    new_pm = np.zeros(L)
                    for out_i, src in enumerate(idx_keep):
                        if src < L:
                            new_llrs.append(llrs_list[src])
                            new_s.append(s_list[src])
                            new_pm[out_i] = pm_dm[src]
                        else:
                            src_l = src - L
                            new_llrs.append(copy.deepcopy(llrs_list[src_l]))
                            new_s.append(copy.deepcopy(s_list[src_l]))
                            new_s[-1][0, bit_idx] = 1 - s_list[src_l][0, bit_idx]
                            new_pm[out_i] = pm_dm[src]
                    llrs_list = new_llrs
                    s_list = new_s
                    pm = new_pm

        best = int(np.argmin(pm))
        u_hat = s_list[best][0, :].astype(int)

        if self.crc_length > 0:
            payload = u_hat[self.info_positions]
            if crc_check(payload, self.crc_length):
                return u_hat, pm[best]
            for d in range(L):
                cand = s_list[d][0, :].astype(int)
                payload = cand[self.info_positions]
                if crc_check(payload, self.crc_length):
                    return cand, pm[d]

        return u_hat, pm[best]
