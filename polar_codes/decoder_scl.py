"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _is_frozen,
    _lower_llr,
    _upper_llr,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n, llr_ch_br):
      self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
      self.B = np.zeros((N, n + 1), dtype=np.int8)
      self.L[:, 0] = llr_ch_br.copy()
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.decode_order = [int(self.br[i]) for i in range(N)]
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _clone_path(self, src):
        dst = _Path(self.N, self.n, src.L[:, 0])
        dst.L[:] = src.L
        dst.B[:] = src.B
        dst.pm = src.pm
        dst.u_hat[:] = src.u_hat
        return dst

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_br = llr_ch[self.br]
        paths = [_Path(self.N, self.n, llr_br)]

        for l in self.decode_order:
            candidates = []
            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if _is_frozen(self.frozen_bits, l):
                    pen = self._penalty(cur_llr, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._penalty(cur_llr, bit)
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
