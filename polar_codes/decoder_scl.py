"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _reorder_channel_llrs,
    _update_bits,
    _update_llrs,
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(info_bits, crc_length):
    if crc_length == 8:
        poly = CRC8_POLY
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        return reg
    if crc_length == 16:
        poly = CRC16_POLY
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        return reg
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        return pm + (0.0 if u == hard else abs(llr))

    def decode(self, llr_ch):
        llr_ch = _reorder_channel_llrs(llr_ch)
        N = self.N
        n = self.n

        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch

        paths = [(0.0, L0, B0, np.zeros(N, dtype=int))]

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for pm, L, B, u_hat in paths:
                _update_llrs(L, B, l, n)
                llr = L[l, n]

                if self.frozen_bits[l]:
                    pm0 = self._pm_update(pm, llr, 0)
                    Bn = B.copy()
                    Ln = L.copy()
                    u_new = u_hat.copy()
                    u_new[l] = 0
                    Bn[l, n] = 0
                    _update_bits(Bn, l, n)
                    new_paths.append((pm0, Ln, Bn, u_new))
                else:
                    for u in (0, 1):
                        pm_u = self._pm_update(pm, llr, u)
                        Bn = B.copy()
                        Ln = L.copy()
                        u_new = u_hat.copy()
                        u_new[l] = u
                        Bn[l, n] = u
                        _update_bits(Bn, l, n)
                        new_paths.append((pm_u, Ln, Bn, u_new))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        best_pm = float("inf")
        best_u = paths[0][3]
        crc_candidates = []

        for pm, _, _, u_hat in paths:
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_candidates.append((pm, u_hat))
            if pm < best_pm:
                best_pm = pm
                best_u = u_hat.copy()

        if crc_candidates:
            crc_candidates.sort(key=lambda x: x[0])
            return crc_candidates[0][1], crc_candidates[0][0]

        return best_u, best_pm


def _precompute_sc_indices_v2(N):
    n = int(math.log2(N))
    llr_layer_vec = [
        list(range(n - active_llr_level(phi, n), n)) for phi in range(N)
    ]
    bit_layer_vec = []
    for phi in range(N):
        if phi < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(
                list(range(n, n - active_bit_level(phi, n), -1))
            )
    return None, llr_layer_vec, bit_layer_vec
