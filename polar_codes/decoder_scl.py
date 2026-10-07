"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, _B_check, _s_updater, _Li


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
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
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


def _permute_llr(llr_ch):
    br = bit_reversal_permutation(len(llr_ch))
    return np.asarray(llr_ch, dtype=np.float64)[br]


class SCLDecoder:
    """SCL 译码器（路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = _permute_llr(llr_ch)
        L = self.list_size
        N, n = self.N, self.n

        llrs = [-np.inf * np.ones((n + 1, N), dtype=np.float64) for _ in range(L)]
        for l in range(L):
            llrs[l][n, :] = llr_ch
        s_paths = [-np.ones((n + 1, N), dtype=np.int8) for _ in range(L)]
        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0

        for ii in range(N):
            if self.frozen_bits[ii]:
                for l in range(L):
                    llrs[l][0, ii] = _Li(0, ii, llrs[l], s_paths[l], n)
                    s_paths[l][0, ii] = 0
                    pm[l] += max(0.0, -llrs[l][0, ii]) if llrs[l][0, ii] < 0 else 0.0
            else:
                dm = np.zeros(L, dtype=np.float64)
                for l in range(L):
                    llrs[l][0, ii] = _Li(0, ii, llrs[l], s_paths[l], n)
                    s_paths[l][0, ii] = 1 if llrs[l][0, ii] < 0 else 0
                    dm[l] = abs(llrs[l][0, ii])

                if L > 1:
                    pm_dm = np.concatenate([pm, pm + dm])
                    idx_sort = np.argsort(pm_dm)[:L]
                    new_llrs = []
                    new_s = []
                    new_pm = np.zeros(L, dtype=np.float64)
                    for k, idx in enumerate(idx_sort):
                        src = idx % L
                        bit_flip = idx >= L
                        ll_copy = llrs[src].copy()
                        s_copy = s_paths[src].copy()
                        if bit_flip:
                            s_copy[0, ii] = 1 - s_copy[0, ii]
                            new_pm[k] = pm_dm[idx]
                        else:
                            new_pm[k] = pm[src]
                        new_llrs.append(ll_copy)
                        new_s.append(s_copy)
                    llrs = new_llrs
                    s_paths = new_s
                    pm = new_pm

        best_l = int(np.argmin(pm))
        u_hat = s_paths[best_l][0, :].astype(int)

        if self.crc_length > 0:
            passed = []
            for l in range(L):
                cand = s_paths[l][0, :].astype(int)
                payload = cand[self.info_indices]
                if len(payload) >= self.crc_length and crc_check(payload, self.crc_length):
                    passed.append((pm[l], cand))
            if passed:
                passed.sort(key=lambda x: x[0])
                u_hat = passed[0][1]
                return u_hat, passed[0][0]

        return u_hat, pm[best_l]
