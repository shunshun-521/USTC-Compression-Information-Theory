"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _prepare_channel_llrs,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _frozen_set_from_mask,
)


_CRC8_GEN = 0x107   # x^8 + x^2 + x + 1
_CRC16_GEN = 0x11021  # CRC-16-IBM


def _gf2_crc_remainder(bits, gen, width):
    """GF(2) 多项式长除求 CRC 余数（MSB first）。"""
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << width):
            reg ^= gen
    return reg & ((1 << width) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        gen, width = _CRC8_GEN, 8
    elif crc_length == 16:
        gen, width = _CRC16_GEN, 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _gf2_crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), gen, width
    )
    crc_bits = np.array(
        [(remainder >> (width - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        gen, width = _CRC8_GEN, 8
    elif crc_length == 16:
        gen, width = _CRC16_GEN, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _gf2_crc_remainder(bits, gen, width) == 0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = _frozen_set_from_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        L[:, 0] = llr_ch
        return {
            "L": L,
            "B": B,
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=int),
        }

    def _copy_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
        }

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        n = self.n
        N = self.N

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )
        return L[l, n]

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path["B"]
        n = self.n
        N = self.N
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = _prepare_channel_llrs(llr_ch)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                llr0 = self._update_llrs(path, l)

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    if llr0 < 0:
                        new_path["pm"] += abs(llr0)
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        expected = 0 if llr0 >= 0 else 1
                        if bit != expected:
                            new_path["pm"] += abs(llr0)
                        new_path["u_hat"][l] = bit
                        new_path["B"][l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            crc_paths = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(path)
            pool = crc_paths if crc_paths else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
