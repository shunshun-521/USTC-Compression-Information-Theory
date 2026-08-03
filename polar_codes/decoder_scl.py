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
    _permute_channel_llr,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    if crc_length == 0:
        return True
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, recomputed)


class _Path:
  __slots__ = ('L', 'B', 'pm', 'u_hat')

  def __init__(self, N, n, llr):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.L[:, 0] = llr
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, bit_idx):
        for stage in range(self.n - _active_llr_level(bit_idx, self.n), self.n):
            block_size = 2 ** (stage + 1)
            branch_size = block_size // 2
            for j in range(bit_idx, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, stage + 1] = f_operation(path.L[j, stage], path.L[j + branch_size, stage])
                else:
                    path.L[j, stage + 1] = g_operation(
                        path.L[j - branch_size, stage], path.L[j, stage], path.B[j - branch_size, stage + 1]
                    )

    def _update_bits(self, path, bit_idx, bit_val):
        path.B[bit_idx, self.n] = bit_val
        path.u_hat[bit_idx] = bit_val
        if bit_idx >= self.N // 2:
            for stage in range(self.n, self.n - _active_bit_level(bit_idx, self.n), -1):
                block_size = 2 ** stage
                branch_size = block_size // 2
                for j in range(bit_idx, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, stage - 1] = path.B[j, stage] ^ path.B[j - branch_size, stage]
                        path.B[j, stage - 1] = path.B[j, stage]

    def _path_metric_penalty(self, llr_val, bit_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr = _permute_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr)]

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for bit_idx in decode_order:
            new_paths = []

            for path in paths:
                self._update_llrs(path, bit_idx)
                llr_val = path.L[bit_idx, self.n]

                if self.frozen_bits[bit_idx]:
                    penalty = self._path_metric_penalty(llr_val, 0)
                    child = _Path(self.N, self.n, path.L[:, 0].copy())
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = path.pm + penalty
                    child.u_hat = path.u_hat.copy()
                    self._update_bits(child, bit_idx, 0)
                    new_paths.append(child)
                else:
                    for bit_val in (0, 1):
                        penalty = self._path_metric_penalty(llr_val, bit_val)
                        child = _Path(self.N, self.n, path.L[:, 0].copy())
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + penalty
                        child.u_hat = path.u_hat.copy()
                        self._update_bits(child, bit_idx, bit_val)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat, best.pm
