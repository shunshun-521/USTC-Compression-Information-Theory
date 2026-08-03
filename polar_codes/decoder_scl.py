"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def _poly_generator(crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    return np.array([(poly >> i) & 1 for i in range(crc_length)] + [1], dtype=int)


def _crc_remainder_gf2(bits, crc_length):
    bits = np.asarray(bits, dtype=int)
    gen = _poly_generator(crc_length)
    reg = bits.copy()
    L = len(bits) - crc_length if len(bits) > crc_length else len(bits)
    if len(bits) == L:
        reg = np.concatenate([reg, np.zeros(crc_length, dtype=int)])
        msg_len = L
    else:
        msg_len = L
    for i in range(msg_len):
        if reg[i]:
            reg[i:i + crc_length + 1] ^= gen
    return reg[msg_len:msg_len + crc_length]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _crc_remainder_gf2(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    gen = _poly_generator(crc_length)
    reg = bits.copy()
    L = len(bits) - crc_length
    for i in range(L):
        if reg[i]:
            reg[i:i + crc_length + 1] ^= gen
    return np.all(reg[L:] == 0)


class _Path:
    __slots__ = ('L', 'B', 'pm', 'active')

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        start = self.n - _active_llr_level(l, self.n)
        for s in range(start, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        end = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    @staticmethod
    def _copy_path(path):
        new_path = _Path.__new__(_Path)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.active = True
        return new_path

    def decode(self, llr_ch):
        llr_work = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        paths = [_Path(self.N, self.n, llr_work.copy())]
        phase_order = [self.rev[i] for i in range(self.N)]

        for l in phase_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path.B[l, self.n] = 0
                    new_path.pm += self._pm_penalty(llr, 0)
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.B[l, self.n] = bit
                        new_path.pm += self._pm_penalty(llr, bit)
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = None
        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.B[:, self.n][~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n], best.pm
