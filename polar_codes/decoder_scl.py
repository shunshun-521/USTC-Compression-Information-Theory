"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _scd_core,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    upper_llr,
    lower_llr,
    f_operation,
    g_operation,
    _frozen_mask,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    reg = 0
    for b in bits:
        reg ^= int(b) << (width - 1)
        for _ in range(8):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        rem = _crc_remainder(info_bits, _CRC8_POLY, 8)
        crc_bits = np.array([(rem >> i) & 1 for i in range(7, -1, -1)], dtype=np.int8)
    elif crc_length == 16:
        rem = _crc_remainder(info_bits, _CRC16_POLY, 16)
        crc_bits = np.array([(rem >> i) & 1 for i in range(15, -1, -1)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        rem = _crc_remainder(bits, _CRC8_POLY, 8)
        return rem == 0
    if crc_length == 16:
        rem = _crc_remainder(bits, _CRC16_POLY, 16)
        return rem == 0
    raise ValueError("crc_length must be 8 or 16")


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制在 L 较小时开销可接受）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_mask(frozen_bits)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen)[0]

    def _init_paths(self, llr_ch):
        p = _Path(self.N, self.n)
        p.L[:, 0] = llr_ch
        return [p]

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        N, n = self.N, self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]
                if np.isnan(cur_llr):
                    cur_llr = 0.0

                if self.frozen[l]:
                    bit = 0
                    penalty = abs(cur_llr) if cur_llr < 0 else 0.0
                    new_pm = path.pm + penalty
                    candidates.append((new_pm, path, bit))
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and cur_llr >= 0) or (bit == 1 and cur_llr < 0) else abs(
                            cur_llr
                        )
                        candidates.append((path.pm + penalty, path, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.L_size]

            new_paths = []
            for pm, parent, bit in selected:
                child = _Path(self.N, self.n)
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.pm = pm
                child.u_hat = parent.u_hat.copy()
                child.B[l, self.n] = bit
                child.u_hat[l] = bit
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p.u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = paths[0]
        return best.u_hat.astype(int), best.pm
