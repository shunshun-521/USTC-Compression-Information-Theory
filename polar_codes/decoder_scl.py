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
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
    prepare_channel_llr,
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
                break
        if crc_length == 16:
            if reg & (1 << 15):
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 译码器 ====================


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n, llr_ch):
      self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
      self.B = np.full((N, n + 1), np.nan)
      self.L[:, 0] = llr_ch
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)

  def copy(self):
      p = _Path.__new__(_Path)
      p.L = self.L.copy()
      p.B = self.B.copy()
      p.pm = self.pm
      p.u_hat = self.u_hat.copy()
      return p


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        llr_ch = prepare_channel_llr(np.asarray(llr_ch, dtype=np.float64), self.N)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    penalty = self._path_metric_penalty(llr, 0)
                    new_path.pm += penalty
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        penalty = self._path_metric_penalty(llr, bit)
                        new_path.pm += penalty
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        # 选择最优路径
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best.u_hat.astype(int), best.pm
