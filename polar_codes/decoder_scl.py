"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _prepare_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    if crc_length == 16:
        return np.array([int(b) for b in format(CRC16_POLY, '017b')], dtype=int)
    raise ValueError(f'Unsupported CRC length: {crc_length}')


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    r = crc_length
    reg = np.zeros(r, dtype=int)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    r = crc_length
    reg = np.zeros(r, dtype=int)
    for bit in bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return np.all(reg == 0)


class _Path:
    __slots__ = ('pm', 'L', 'B', 'parent')

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.parent = None

    def copy(self):
        child = _Path.__new__(_Path)
        child.pm = self.pm
        child.L = self.L
        child.B = self.B
        child.parent = self
        return child

    def materialize(self):
        if self.parent is not None:
            self.L = self.parent.L.copy()
            self.B = self.parent.B.copy()
            self.parent = None


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        path.materialize()
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    if top_bit == 0:
                        path.L[j, s + 1] = path.L[j, s] + path.L[j - branch_size, s]
                    else:
                        path.L[j, s + 1] = path.L[j, s] - path.L[j - branch_size, s]
        return path.L[l, self.n]

    def _update_bits(self, path, l, u_bit):
        path.materialize()
        path.B[l, self.n] = u_bit
        if l >= self.N / 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                        path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        preferred = 0 if llr >= 0 else 1
        return 0.0 if u_bit == preferred else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        llr = _prepare_llr(np.asarray(llr_ch, dtype=np.float64), self.N)
        paths = [_Path(self.N, self.n, llr)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                llr_val = self._update_llrs(path, l)
                if l in self.frozen:
                    child = path.copy()
                    child.pm += self._pm_penalty(llr_val, 0)
                    self._update_bits(child, l, 0)
                    candidates.append(child)
                else:
                    for u_bit in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_penalty(llr_val, u_bit)
                        self._update_bits(child, l, u_bit)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        for path in paths:
            path.materialize()

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.B[:, self.n][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int), best.pm
