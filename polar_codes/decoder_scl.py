"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import boxplus, _bit_reversed_index, _active_llr_level, _active_bit_level
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
  CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0

    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    for _ in range(crc_length):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否包含正确的 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat', 'copied')

    def __init__(self, n, N, parent=None):
        self.copied = parent is None
        if parent is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self.pm = 0.0
            self.u_hat = np.zeros(N, dtype=int)
        else:
            self.L = parent.L
            self.B = parent.B
            self.pm = parent.pm
            self.u_hat = parent.u_hat.copy()
            self.copied = False

    def ensure_copy(self):
        if not self.copied:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.copied = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = boxplus(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    if top_bit == 0:
                        path.L[j, s + 1] = path.L[j, s] + path.L[j - branch_size, s]
                    else:
                        path.L[j, s + 1] = path.L[j, s] - path.L[j - branch_size, s]

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        n = self.n
        N = self.N
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[bit_reversal_permutation(self.N)]
        n = self.n
        N = self.N

        paths = [_Path(n, N)]
        paths[0].L[:, 0] = llr

        for i in range(N):
            l = _bit_reversed_index(i, n)
            candidates = []

            for path in paths:
                path.ensure_copy()
                self._update_llrs(path, l)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    u = 0
                    path.pm += self._pm_penalty(llr_val, u)
                    path.u_hat[l] = u
                    path.B[l, n] = u
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        new_path = _Path(n, N, parent=path)
                        new_path.ensure_copy()
                        new_path.pm = path.pm + self._pm_penalty(llr_val, u)
                        new_path.u_hat[l] = u
                        new_path.B[l, n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        crc_passed = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_passed.append(path)

        best = min(crc_passed, key=lambda p: p.pm) if crc_passed else min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
