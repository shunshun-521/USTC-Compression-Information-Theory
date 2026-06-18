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
    _reorder_channel_llr,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & (1 << (crc_length - 1)):
                    reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
                else:
                    reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


# ==================== SCL 译码器 ====================


class _Path:
  __slots__ = ("L", "B", "pm", "active")

  def __init__(self, N, n):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.full((N, n + 1), np.nan)
    self.pm = 0.0
    self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_llr(self, path, l):
        _update_llrs_path(path.L, path.B, l, self.n)

    def _path_bits(self, path, l, bit):
        _update_bits_path(path.B, l, self.n, bit)

    def _branch_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _reorder_channel_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n) for _ in range(1)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for p_idx, path in enumerate(paths):
                if not path.active:
                    continue
                self._path_llr(path, l)
                llr = path.L[l, self.n]
                if np.isnan(llr):
                    llr = 0.0

                if self.frozen_bits[l]:
                    penalty = self._branch_penalty(llr, 0)
                    candidates.append((path.pm + penalty, p_idx, 0, False))
                else:
                    for bit in (0, 1):
                        penalty = self._branch_penalty(llr, bit)
                        candidates.append((path.pm + penalty, p_idx, bit, True))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            parent_map = {}

            for pm, p_idx, bit, _ in candidates:
                if len(new_paths) >= self.list_size:
                    break
                key = (p_idx, bit)
                if key in parent_map:
                    continue
                parent_map[key] = True
                parent = paths[p_idx]
                child = _Path(self.N, self.n)
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.pm = pm
                self._path_llr(child, l)
                child.B[l, self.n] = bit
                self._path_bits(child, l, bit)
                new_paths.append(child)

            paths = new_paths

        best = None
        if self.crc_length > 0:
            for path in paths:
                u_hat = path.B[:, self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best is None or path.pm < best.pm:
                        best = path
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm


def _update_llrs_path(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits_path(B, l, n, bit):
    B[l, n] = bit
    if l < len(B) // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]
