"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    _upper_llr,
    _lower_llr,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    f_operation,
    g_operation,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数。"""
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
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


# ==================== SCL 译码器 ====================


class _Path:
  """单条译码路径（Lazy Copy）。"""

  __slots__ = ("L", "B", "pm", "u_hat", "active")

  def __init__(self, N, n, llr):
      self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
      self.B = np.full((N, n + 1), np.nan)
      self.L[:, 0] = llr
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)
      self.active = True


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

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
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

    def _pm_penalty(self, llr, bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|。"""
        preferred = 0 if llr >= 0 else 1
        return 0.0 if bit == preferred else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr = llr_ch[rev]

        paths = [_Path(self.N, self.n, llr)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                if not path.active:
                    continue
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = self._pm_penalty(cur_llr, 0)
                    path.pm += penalty
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    candidates.append((path.pm, pidx, None))
                else:
                    for bit in (0, 1):
                        new_pm = path.pm + self._pm_penalty(cur_llr, bit)
                        candidates.append((new_pm, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            used = set()

            for pm, pidx, bit in candidates:
                if len(new_paths) >= self.list_size:
                    break
                src = paths[pidx]
                if bit is None:
                    dst = src
                else:
                    key = (pidx, bit)
                    if key in used:
                        continue
                    used.add(key)
                    dst = _Path(self.N, self.n, llr)
                    dst.L = src.L.copy()
                    dst.B = src.B.copy()
                    dst.pm = pm
                    dst.u_hat = src.u_hat.copy()
                    dst.B[l, self.n] = bit
                    dst.u_hat[l] = bit
                    self._update_bits(dst, l)

                if bit is None:
                    dst.pm = pm
                new_paths.append(dst)

            paths = new_paths

        # 选择最优路径
        best_path = None
        best_pm = float("inf")

        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if path.pm < best_pm:
                        best_pm = path.pm
                        best_path = path

        if best_path is None:
            best_path = min(paths, key=lambda p: p.pm)
            best_pm = best_path.pm

        return best_path.u_hat.copy(), best_pm
