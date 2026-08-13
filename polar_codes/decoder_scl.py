"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import _compute_llr, _s_updater, _b_check


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L = self.list_size

        paths_llrs = [np.full((n + 1, N), -np.inf, dtype=np.float64) for _ in range(L)]
        paths_bits = [np.full((n + 1, N), -1, dtype=np.int32) for _ in range(L)]
        for p in paths_llrs:
            p[n, :] = llr_ch

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0

        for phi in range(N):
            dm = np.zeros(L, dtype=np.float64)

            if self.frozen_bits[phi]:
                for dd in range(L):
                    llr = _compute_llr(0, phi, paths_llrs[dd], paths_bits[dd])
                    paths_bits[dd][0, phi] = 0
                    paths_llrs[dd][0, phi] = np.inf
                    pm[dd] += -llr * (llr < 0)
            else:
                for dd in range(L):
                    llr = _compute_llr(0, phi, paths_llrs[dd], paths_bits[dd])
                    bit = 0 if llr >= 0 else 1
                    paths_bits[dd][0, phi] = bit
                    dm[dd] = abs(llr)

                if L > 1:
                    pm_dm = np.concatenate([pm, pm + dm])
                    idx_sort = np.argsort(pm_dm)
                    idx_low = idx_sort[:L]
                    idx_high = idx_sort[L:]

                    idx_min_low = idx_low[idx_low >= L] - L
                    idx_min_up = idx_high[idx_high < L]

                    for low_i, up_i in zip(idx_min_low, idx_min_up):
                        paths_llrs[up_i] = paths_llrs[low_i].copy()
                        paths_bits[up_i] = paths_bits[low_i].copy()
                        paths_bits[up_i][0, phi] = 1 - paths_bits[low_i][0, phi]
                        pm[up_i] = pm_dm[low_i + L]

        u_candidates = [paths_bits[dd][0, :].copy() for dd in range(L)]

        if self.crc_length > 0:
            valid = []
            for dd, u_hat in enumerate(u_candidates):
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm[dd], u_hat))
            if valid:
                best_pm, best_u = min(valid, key=lambda x: x[0])
            else:
                best_idx = int(np.argmin(pm))
                best_u = u_candidates[best_idx]
                best_pm = pm[best_idx]
        else:
            best_idx = int(np.argmin(pm))
            best_u = u_candidates[best_idx]
            best_pm = pm[best_idx]

        return best_u.astype(int), float(best_pm)
