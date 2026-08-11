"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _B_check,
    _s_updater,
    _compute_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        if self.frozen_bits.dtype != bool:
            self.frozen_bits = self.frozen_bits.astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size
        frozen = self.frozen_bits

        llrs_list = []
        bits_list = []
        for _ in range(L):
            llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
            llrs[n, :] = llr_ch
            llrs_list.append(llrs)
            bits_list.append(-np.ones((n + 1, N), dtype=int))

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0
        u_hat_paths = [np.zeros(N, dtype=int) for _ in range(L)]

        for i in range(N):
            dm = np.zeros(L, dtype=np.float64)

            for dd in range(L):
                llrs = llrs_list[dd]
                bits = bits_list[dd]
                llrs[0, i] = _compute_llr(0, i, llrs, bits)

                if frozen[i]:
                    u_hat_paths[dd][i] = 0
                    bits[0, i] = 0
                    pm[dd] += -llrs[0, i] * (llrs[0, i] < 0)
                else:
                    u_hat_paths[dd][i] = 1 if llrs[0, i] < 0 else 0
                    bits[0, i] = u_hat_paths[dd][i]
                    dm[dd] = abs(llrs[0, i])

            if not frozen[i] and L > 1:
                pm_dm = np.concatenate([pm, pm + dm])
                idx_sort = np.argsort(pm_dm)
                idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
                idx_min_up = idx_sort[L:][idx_sort[L:] < L]

                for bb in range(len(idx_min_low)):
                    low, up = idx_min_low[bb], idx_min_up[bb]
                    llrs_list[up] = llrs_list[low].copy()
                    bits_list[up] = bits_list[low].copy()
                    u_hat_paths[up] = u_hat_paths[low].copy()
                    u_hat_paths[up][i] = 1 - u_hat_paths[low][i]
                    bits_list[up][0, i] = u_hat_paths[up][i]
                    pm[up] = pm_dm[low + L]

        best_idx = self._select_best_index(u_hat_paths, pm)
        return u_hat_paths[best_idx].copy(), pm[best_idx]

    def _select_best_index(self, paths, pm):
        if self.crc_length > 0:
            crc_pass = []
            for idx, u_hat in enumerate(paths):
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(idx)
            if crc_pass:
                return min(crc_pass, key=lambda i: pm[i])
        return int(np.argmin(pm))


def scl_decode_equivalent_sc(llr_ch, frozen_bits):
    """L=1 的 SCL 应等价于 SC。"""
    scl = SCLDecoder(len(llr_ch), frozen_bits, list_size=1, crc_length=0)
    u_scl, _ = scl.decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return u_scl, u_sc
