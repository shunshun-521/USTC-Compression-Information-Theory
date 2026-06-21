"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    channel_llr_to_decode,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
  """
  计算 CRC 校验位并附加到信息比特后。
  CRC-8: 0x07, CRC-16: 0x8005
  """
  poly = _crc_poly(crc_length)
  reg = 0
  for bit in np.asarray(info_bits, dtype=int):
    reg ^= int(bit) << (crc_length - 1)
    for _ in range(8):
      if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
      else:
        reg = (reg << 1) & ((1 << crc_length) - 1)
  crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
  return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  if crc_length <= 0:
    return True
  bits = np.asarray(bits, dtype=int)
  expected = crc_encode(bits[:-crc_length], crc_length)
  return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "parent", "branch_bit")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.parent = None
        self.branch_bit = 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        return new_path

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_internal = channel_llr_to_decode(llr_ch)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_internal

        for l in self.decode_order:
            for path in paths:
                self._update_llrs(path, l)

            new_paths = []
            if self.frozen_bits[l]:
                for path in paths:
                    llr = path.L[l, self.n]
                    path.pm += self._path_metric_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
            else:
                for path in paths:
                    llr = path.L[l, self.n]
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        child.pm += self._path_metric_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path.B[:, self.n].astype(int)
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
