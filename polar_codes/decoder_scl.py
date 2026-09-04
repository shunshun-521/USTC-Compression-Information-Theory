"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    _update_bits,
    _upper_llr,
    bit_reversed,
    f_operation,
    precompute_sc_indices,
    sc_decode,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（CRC-8: 0x07, CRC-16: 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    payload = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    for bit in payload:
        reg ^= int(bit)
        for _ in range(8):
            if reg & 1:
                reg = (reg >> 1) ^ poly
            else:
                reg >>= 1
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit)
        for _ in range(8):
            if reg & 1:
                reg = (reg >> 1) ^ poly
            else:
                reg >>= 1
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr):
        return {
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=int),
        }

    def _update_llrs_path(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = _upper_llr(
                        path["L"][j, s],
                        path["L"][j + branch_size, s],
                    )
                else:
                    path["L"][j, s + 1] = _lower_llr(
                        path["L"][j, s],
                        path["L"][j - branch_size, s],
                        path["B"][j - branch_size, s + 1],
                    )

    @staticmethod
    def _penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]
        paths[0]["L"][:, 0] = llr_ch

        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []
            for path in paths:
                self._update_llrs_path(path, l)
                llr0 = path["L"][l, self.n]
                bits = [0] if self.frozen_bits[l] else [0, 1]
                for bit in bits:
                    new_path = copy.deepcopy(path)
                    new_path["pm"] += self._penalty(llr0, bit)
                    new_path["B"][l, self.n] = bit
                    new_path["u_hat"][l] = bit
                    _update_bits(new_path["B"], l, self.n, self.N)
                    candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
