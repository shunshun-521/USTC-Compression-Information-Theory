"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    reg = 0
    for b in bits:
        reg ^= int(b) << (width - 1)
        for _ in range(8 if width == 8 else 16):
            if width == 8:
                msb = reg & 0x80
                reg = (reg << 1) & 0xFF
                if msb:
                    reg ^= poly
            else:
                msb = reg & 0x8000
                reg = (reg << 1) & 0xFFFF
                if msb:
                    reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        rem = _crc_remainder(info_bits, _CRC8_POLY, 8)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        rem = _crc_remainder(info_bits, _CRC16_POLY, 16)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        rem = _crc_remainder(bits, _CRC8_POLY, 8)
        return rem == 0
    if crc_length == 16:
        rem = _crc_remainder(bits, _CRC16_POLY, 16)
        return rem == 0
    raise ValueError("crc_length must be 8 or 16")


def _path_metric_penalty(llr, bit):
    """与 SC 一致：LLR>=0 判 0，否则 1；不一致则加 |LLR|"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        start_s = self.n - _active_llr_level(l, self.n)
        for s in range(start_s, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path["B"]
        end_s = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end_s, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """SCL 主译码"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = self.decode_order[i]
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    pen = _path_metric_penalty(llr_val, 0)
                    new_path = self._copy_path(path)
                    new_path["pm"] += pen
                    new_path["B"][l, self.n] = 0
                    new_path["u_hat"][l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pen = _path_metric_penalty(llr_val, bit)
                        new_path = self._copy_path(path)
                        new_path["pm"] += pen
                        new_path["B"][l, self.n] = bit
                        new_path["u_hat"][l] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = []
            for p in paths:
                info_bits = p["u_hat"][info_mask]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].astype(int), best["pm"]
