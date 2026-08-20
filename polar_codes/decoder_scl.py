"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import bit_reversed
from decoder_sc import _upper_llr, _lower_llr, _active_llr_level, _active_bit_level


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_serial(bits, poly, width):
    reg = 0
    mask = (1 << width) - 1
    for bit in bits:
        feedback = ((reg >> (width - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_serial(info_bits, poly, crc_length)
    crc_part = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_part])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_serial(bits, poly, crc_length) == 0


class PathState:
    __slots__ = ('pm', 'L', 'B')

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        L = path.L
        B = path.B
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        if l < N // 2:
            return
        B = path.B
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        N, n = self.N, self.n
        paths = [PathState(N, n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    new_path = PathState(N, n)
                    new_path.pm = path.pm + self._pm_penalty(llr, 0)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.B[l, n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = PathState(N, n)
                        new_path.pm = path.pm + self._pm_penalty(llr, u)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.B[l, n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            passed = [p for p in paths
                      if crc_check(p.B[:, n][self.info_indices], self.crc_length)]
            if passed:
                best = min(passed, key=lambda p: p.pm)

        return best.B[:, n].astype(int).copy(), best.pm
