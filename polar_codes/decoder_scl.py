"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _prepare_llr,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    steps = 8 if crc_length == 8 else 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(steps):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits 是否通过 CRC 校验。
    """
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    steps = 8 if crc_length == 8 else 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(steps):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """
    SCL 译码器。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        return pm + (0.0 if u == hard else abs(llr))

    def _update_llrs(self, L, B, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    @staticmethod
    def _update_bits(B, l, n, N):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。返回 (u_hat, pm)
        """
        llr_ch = _prepare_llr(llr_ch)
        N, n, L_size = self.N, self.n, self.list_size

        paths = [
            {
                "L": np.zeros((N, n + 1)),
                "B": np.zeros((N, n + 1), dtype=int),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for l in [_bit_reversed(i, n) for i in range(N)]:
            new_paths = []
            for path in paths:
                L_arr = path["L"].copy()
                B_arr = path["B"].copy()
                self._update_llrs(L_arr, B_arr, l)
                cur_llr = L_arr[l, n]

                if l in self.frozen_set:
                    B_arr[l, n] = 0
                    self._update_bits(B_arr, l, n, N)
                    new_paths.append(
                        {
                            "L": L_arr,
                            "B": B_arr,
                            "pm": self._pm_update(path["pm"], cur_llr, 0),
                        }
                    )
                else:
                    for u_val in (0, 1):
                        La = L_arr.copy()
                        Ba = B_arr.copy()
                        Ba[l, n] = u_val
                        self._update_bits(Ba, l, n, N)
                        new_paths.append(
                            {
                                "L": La,
                                "B": Ba,
                                "pm": self._pm_update(path["pm"], cur_llr, u_val),
                            }
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["B"][:, n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, n].astype(int), best["pm"]
