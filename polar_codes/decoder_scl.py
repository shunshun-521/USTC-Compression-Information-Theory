"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_operation,
    g_operation,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_encode_3gpp(bits, crc_length, poly):
    """3GPP 风格 CRC 编码，返回信息位 + 校验位。"""
    bits = list(np.asarray(bits, dtype=int))
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        feedback = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | bit) & mask
        if feedback:
            reg ^= poly
    for _ in range(crc_length):
        feedback = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.array(bits + crc_bits, dtype=int)


def _crc_check_3gpp(bits, crc_length, poly):
    """3GPP 风格 CRC 校验。"""
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in np.asarray(bits, dtype=int):
        feedback = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | bit) & mask
        if feedback:
            reg ^= poly
    return reg == 0


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_encode_3gpp(info_bits, crc_length, poly)


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC。"""
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_check_3gpp(bits, crc_length, poly)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _new_path(self, llr_ch):
        path = {
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=int),
            "pm": 0.0,
        }
        path["L"][:, 0] = llr_ch
        return path

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = f_operation(
                        np.array([path["L"][j, s]]),
                        np.array([path["L"][j + branch_size, s]]),
                    )[0]
                else:
                    path["L"][j, s + 1] = g_operation(
                        np.array([path["L"][j - branch_size, s]]),
                        np.array([path["L"][j, s]]),
                        np.array([path["B"][j - branch_size, s + 1]]),
                    )[0]

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path["B"][j - branch_size, s - 1] = (
                        path["B"][j, s] ^ path["B"][j - branch_size, s]
                    )
                    path["B"][j, s - 1] = path["B"][j, s]

    def _penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]

                if l in self.frozen_set:
                    child = copy.deepcopy(path)
                    child["pm"] += self._penalty(llr, 0)
                    child["B"][l, self.n] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = copy.deepcopy(path)
                        child["pm"] += self._penalty(llr, bit)
                        child["B"][l, self.n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u = path["B"][:, self.n].astype(int)
                if crc_check(u[self.info_indices], self.crc_length):
                    valid.append(path)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p["pm"])
        u_hat = best["B"][:, self.n].astype(int)
        return u_hat, best["pm"]
