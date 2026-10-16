"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import (
    sc_decode,
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ CRC8_POLY) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        reg = 0xFFFF
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ CRC16_POLY) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 LLR/比特状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _llr_at_phase(self, path, phase):
        l = _bit_reversed(phase, self.n)
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path["L"][j, s + 1] = f_operation(path["L"][j, s], path["L"][j + branch_size, s])
                else:
                    path["L"][j, s + 1] = g_operation(
                        path["L"][j, s],
                        path["L"][j - branch_size, s],
                        int(path["B"][j - branch_size, s + 1]),
                    )
        return path["L"][l, self.n]

    def _update_bits(self, path, phase):
        l = _bit_reversed(phase, self.n)
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path["B"][j - branch_size, s - 1] = int(path["B"][j, s]) ^ int(
                        path["B"][j - branch_size, s]
                    )
                    path["B"][j, s - 1] = path["B"][j, s]

    def _new_path(self, llr_ch):
        path = {
            "pm": 0.0,
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan),
            "u_hat": np.zeros(self.N, dtype=np.int8),
        }
        path["L"][:, 0] = llr_ch
        return path

    def _pm_update(self, pm, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        if u_bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phase in range(self.N):
            l = _bit_reversed(phase, self.n)
            new_paths = []
            for path in paths:
                llr = self._llr_at_phase(path, phase)
                if self.frozen_bits[l]:
                    child = copy.deepcopy(path)
                    child["pm"] = self._pm_update(path["pm"], llr, 0)
                    child["u_hat"][l] = 0
                    child["B"][l, self.n] = 0
                    self._update_bits(child, phase)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = copy.deepcopy(path)
                        child["pm"] = self._pm_update(path["pm"], llr, u_bit)
                        child["u_hat"][l] = u_bit
                        child["B"][l, self.n] = u_bit
                        self._update_bits(child, phase)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            passed = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_positions], self.crc_length)
            ]
            if passed:
                best = min(passed, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
