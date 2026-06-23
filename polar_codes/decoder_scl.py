"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation,
    g_operation,
    sc_decode,
)
from encoder import bit_reversal_permutation


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_update(reg, bit, poly, crc_length):
    reg ^= int(bit) << (crc_length - 1)
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for _ in range(8):
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    info = bits[:-crc_length]
    expected_crc = bits[-crc_length:]
    reg = 0
    for bit in info:
        reg = _crc_update(reg, bit, poly, crc_length)
    actual_crc = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.array_equal(actual_crc, expected_crc)


class Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _llr_penalty(self, llr_val, u):
        u_hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == u_hard else abs(llr_val)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
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
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        path = Path(self.N, self.n)
        path.L[:, 0] = llr_ch[self.br]
        paths = [path]

        for i in range(self.N):
            l = self.br[i]
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._clone(path)
                    new_path.pm += self._llr_penalty(cur_llr, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._clone(path)
                        new_path.pm += self._llr_penalty(cur_llr, u)
                        new_path.u_hat[l] = u
                        new_path.B[l, self.n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if not crc_check(info_bits, self.crc_length):
                    continue
            if best_pm is None or path.pm < best_pm:
                best_pm = path.pm
                best_crc = path

        if best_crc is not None:
            return best_crc.u_hat.copy(), best_crc.pm

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    @staticmethod
    def _clone(path):
        new_path = Path(path.L.shape[0], path.L.shape[1] - 1)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path
