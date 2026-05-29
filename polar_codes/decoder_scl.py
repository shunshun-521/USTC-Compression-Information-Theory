"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
    _frozen_index_set,
)


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位计算（信息比特 + CRC）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组，分裂时复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = _frozen_index_set(frozen_bits)
        self.info_indices = sorted(set(range(N)) - self.frozen_set)

    def _clone_path(self, path):
        new_p = Path(self.N, self.n)
        new_p.L = path.L.copy()
        new_p.B = path.B.copy()
        new_p.pm = path.pm
        new_p.u_hat = path.u_hat.copy()
        return new_p

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = lower_llr(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -50.0, 50.0)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    path.pm += self._pm_penalty(llr_bit, 0)
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p2 = self._clone_path(path)
                        p2.u_hat[l] = u_bit
                        p2.B[l, self.n] = u_bit
                        p2.pm += self._pm_penalty(llr_bit, u_bit)
                        self._update_bits(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                valid.sort(key=lambda p: p.pm)
                best = valid[0]

        return best.u_hat.astype(int), best.pm
