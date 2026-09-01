"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


CRC_DIVISORS = {
    8: [1, 0, 0, 0, 0, 0, 1, 1, 1],
    16: [1] + [0] * 14 + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    div = CRC_DIVISORS[crc_length]
    reg = list(info_bits.astype(int)) + [0] * crc_length
    for i in range(len(info_bits)):
        if reg[i]:
            for j in range(len(div)):
                reg[i + j] ^= div[j]
    crc_bits = np.array(reg[-crc_length:], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    bits = np.asarray(bits, dtype=np.int8)
    div = CRC_DIVISORS[crc_length]
    reg = list(bits.astype(int))
    for i in range(len(bits) - crc_length):
        if reg[i]:
            for j in range(len(div)):
                reg[i + j] ^= div[j]
    return all(x == 0 for x in reg[-crc_length:])


class _Path:
  __slots__ = ('L', 'B', 'pm', 'active')

  def __init__(self, N, n, llr_ch):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    br = bit_reversal_permutation(N)
    self.L[:, 0] = llr_ch[br]
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.pm = 0.0
    self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, path, l):
        B = path.B
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = self._pm_penalty(llr, 0)
                    path.pm += penalty
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, llr_ch)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best_any = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            for path in sorted(paths, key=lambda p: p.pm):
                u_hat = path.B[:, self.n].astype(int)
                info_bits = u_hat[info_mask]
                if crc_check(info_bits, self.crc_length):
                    best_crc = path
                    break

        chosen = best_crc if best_crc is not None else best_any
        return chosen.B[:, self.n].astype(int), chosen.pm
