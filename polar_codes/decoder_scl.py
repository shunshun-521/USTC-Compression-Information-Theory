"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import INF, _li, _s_updater, f_operation, g_operation, precompute_sc_indices

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L = self.list_size

        llrs_list = [np.full((n + 1, N), -INF, dtype=np.float64) for _ in range(L)]
        s_list = [-np.ones((n + 1, N), dtype=np.int8) for _ in range(L)]
        for d in range(L):
            llrs_list[d][n] = llr_ch.copy()
        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0

        for phi in range(N):
            metrics = []
            for d in range(L):
                if pm[d] == np.inf:
                    continue
                llr_bit = _li(0, phi, llrs_list[d], s_list[d])
                if self.frozen_bits[phi]:
                    bit = 0
                    pm_new = pm[d] + (abs(llr_bit) if llr_bit < 0 else 0.0)
                    metrics.append((pm_new, d, bit))
                else:
                    for bit in (0, 1):
                        pm_new = pm[d] + (0.0 if (bit == 0 and llr_bit >= 0) or (bit == 1 and llr_bit < 0) else abs(llr_bit))
                        metrics.append((pm_new, d, bit))

            metrics.sort(key=lambda x: x[0])
            selected = metrics[:L]

            new_llrs = []
            new_s = []
            new_pm = np.full(L, np.inf, dtype=np.float64)

            for slot, (pm_val, parent, bit) in enumerate(selected):
                llrs = llrs_list[parent].copy()
                s_arr = s_list[parent].copy()
                llrs[0, phi] = _li(0, phi, llrs, s_arr)
                s_arr[0, phi] = bit
                new_llrs.append(llrs)
                new_s.append(s_arr)
                new_pm[slot] = pm_val

            while len(new_llrs) < L:
                new_llrs.append(np.full((n + 1, N), -INF, dtype=np.float64))
                new_s.append(-np.ones((n + 1, N), dtype=np.int8))

            llrs_list = new_llrs
            s_list = new_s
            pm = new_pm

        paths = []
        for d in range(L):
            if pm[d] == np.inf:
                continue
            u_hat = s_list[d][0].copy()
            u_hat[self.frozen_bits] = 0
            paths.append((pm[d], u_hat))

        if not paths:
            return np.zeros(N, dtype=np.int8), 0.0

        if self.crc_length > 0:
            crc_paths = [
                (p, u) for p, u in paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if crc_paths:
                pm_best, u_best = min(crc_paths, key=lambda x: x[0])
            else:
                pm_best, u_best = min(paths, key=lambda x: x[0])
        else:
            pm_best, u_best = min(paths, key=lambda x: x[0])

        return u_best, pm_best
