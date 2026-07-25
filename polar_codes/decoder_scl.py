"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _frozen_mask_to_mcba1n_set,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.br = bit_reversal_permutation(N)
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.info_idx = np.where(~frozen_bits)[0]
        else:
            self.info_idx = np.where(frozen_bits.astype(int) == 0)[0]
        self.frozen_set = _frozen_mask_to_mcba1n_set(frozen_bits, self.br)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_update(self, pm, llr, bit):
        penalty = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def _update_bits(self, B, l, n):
        if l < self.N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int32),
            "pm": 0.0,
            "u": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phase in range(N):
            l = _bit_reversed(phase, n)
            new_paths = []

            for path in paths:
                L, B, pm, u = path["L"], path["B"], path["pm"], path["u"]

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                            )

                llr_bit = L[l, n]

                if l in self.frozen_set:
                    pm_new = self._path_metric_update(pm, llr_bit, 0)
                    B[l, n] = 0
                    u[l] = 0
                    self._update_bits(B, l, n)
                    new_paths.append({"L": L, "B": B, "pm": pm_new, "u": u.copy()})
                else:
                    for bit in (0, 1):
                        L_copy = L.copy()
                        B_copy = B.copy()
                        u_copy = u.copy()
                        pm_new = self._path_metric_update(pm, llr_bit, bit)
                        B_copy[l, n] = bit
                        u_copy[l] = bit
                        self._update_bits(B_copy, l, n)
                        new_paths.append({
                            "L": L_copy, "B": B_copy, "pm": pm_new, "u": u_copy,
                        })

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p["u"][self.br]
                payload = u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"][self.br], best["pm"]
