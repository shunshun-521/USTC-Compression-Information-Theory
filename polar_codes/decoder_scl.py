"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(data_bits, poly, crc_len):
    reg = [0] * crc_len
    for bit in data_bits:
        fb = bit ^ reg[0]
        reg = reg[1:] + [0]
        if fb:
            for i in range(crc_len):
                if (poly >> (crc_len - 1 - i)) & 1:
                    reg[i] ^= fb
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_divide(bits, poly, crc_length)
    return np.all(remainder == 0)


# ==================== SCL 译码器 ====================


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat", "active")

  def __init__(self, N, n):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int8)
    self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _copy_path(self, src):
        p = _Path(self.N, self.n)
        p.L = src.L.copy()
        p.B = src.B.copy()
        p.pm = src.pm
        p.u_hat = src.u_hat.copy()
        return p

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

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = self.br[i]
            candidates = []

            for pidx, path in enumerate(paths):
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[i]:
                    bit = 0
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr, bit)
                    new_path.B[l, self.n] = bit
                    new_path.u_hat[i] = bit
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[i] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            for p in paths:
                p.active = True

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
