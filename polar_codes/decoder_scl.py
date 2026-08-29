"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _hard_decision,
    _update_bits,
    _update_llrs,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(8 if width == 8 else 1):
            if width == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                break
        if width == 16:
            if reg & (1 << 15):
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        rem = _crc_remainder(info_bits, _CRC8_POLY, 8)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        reg = 0
        poly = _CRC16_POLY
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(1):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        rem = reg
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 译码器 ====================


class _Path:
  __slots__ = ("L", "B", "pm", "active")

  def __init__(self, N, n, llr_ch):
      self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
      self.B = np.full((N, n + 1), np.nan)
      self.L[:, 0] = llr_ch
      self.pm = 0.0
      self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_set = sorted(set(range(N)) - self.frozen_set)

    def _path_metric_penalty(self, llr, bit):
        hard = _hard_decision(llr)
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]

        paths = [_Path(self.N, self.n, llr_ch.copy())]
        decisions = [np.zeros(self.N, dtype=int) for _ in range(self.list_size)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for pidx, path in enumerate(paths):
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    bit = 0
                    pm = path.pm + self._path_metric_penalty(llr, bit)
                    child = _Path(self.N, self.n, path.L[:, 0].copy())
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = pm
                    child.B[l, self.n] = 0
                    decisions[pidx][l] = 0
                    _update_bits(child.B, l, self.n)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._path_metric_penalty(llr, bit)
                        child = _Path(self.N, self.n, path.L[:, 0].copy())
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = pm
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]
            for i, path in enumerate(paths):
                decisions[i][l] = int(path.B[l, self.n])

        best_idx = 0
        if self.crc_length > 0:
            info_bits_order = [l for l in self.info_set]
            valid = []
            for i, path in enumerate(paths):
                u_info = decisions[i][info_bits_order]
                if crc_check(u_info, self.crc_length):
                    valid.append((path.pm, i))
            if valid:
                valid.sort()
                best_idx = valid[0][1]

        u_hat = decisions[best_idx]
        return u_hat, paths[best_idx].pm
