"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    f_operation,
    g_operation,
    precompute_sc_indices,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
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
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, L, B, l):
        n = self.n
        N = self.N
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

    def _update_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L_size = self.N, self.n, self.list_size

        paths = []
        for _ in range(L_size):
            L = np.full((N, n + 1), np.nan, dtype=np.float64)
            B = np.zeros((N, n + 1), dtype=int)
            L[:, 0] = llr_ch.copy()
            paths.append({"pm": 0.0, "L": L, "B": B})

        active = 1

        for l in self.decode_order:
            new_paths = []
            for pidx in range(active):
                path = paths[pidx]
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    pen = self._pm_penalty(llr, 0)
                    np_ = {
                        "pm": path["pm"] + pen,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    np_["B"][l, n] = 0
                    self._update_bits(np_["B"], l)
                    new_paths.append(np_)
                else:
                    for u in (0, 1):
                        pen = self._pm_penalty(llr, u)
                        np_ = {
                            "pm": path["pm"] + pen,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        np_["B"][l, n] = u
                        self._update_bits(np_["B"], l)
                        new_paths.append(np_)

            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[:L_size]
            active = len(paths)

        candidates = []
        for path in paths:
            u_hat = path["B"][:, n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    candidates.append((path["pm"], u_hat))
            else:
                candidates.append((path["pm"], u_hat))

        if candidates:
            pm, u_hat = min(candidates, key=lambda x: x[0])
        else:
            pm, u_hat = min((p["pm"], p["B"][:, n].astype(int)) for p in paths)

        return u_hat, pm
