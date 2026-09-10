"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数。"""
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class PathState:
    """单条 SCL 路径状态（Lazy Copy 友好）。"""

    __slots__ = ("pm", "L", "B", "parent_L", "parent_B")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = [np.zeros((N, n + 1), dtype=np.float64) for _ in range(1)]
        self.B = [np.zeros((N, n + 1), dtype=int) for _ in range(1)]
        self.L[0] = np.zeros((N, n + 1), dtype=np.float64)
        self.B[0] = np.zeros((N, n + 1), dtype=int)
        self.parent_L = None
        self.parent_B = None

    def get_L(self):
        return self.L[0] if self.parent_L is None else self.parent_L

    def get_B(self):
        return self.B[0] if self.parent_B is None else self.parent_B

    def fork(self):
        child = PathState.__new__(PathState)
        child.pm = self.pm
        child.parent_L = self.get_L()
        child.parent_B = self.get_B()
        child.L = [None]
        child.B = [None]
        return child

    def materialize(self):
        if self.L[0] is None:
            self.L[0] = self.get_L().copy()
            self.parent_L = None
        if self.B[0] is None:
            self.B[0] = self.get_B().copy()
            self.parent_B = None


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _init_llr(self, L, llr_ch):
        for i in range(self.N):
            L[_bit_reversed(i, self.n), 0] = llr_ch[i]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.n, self.N)]
        paths[0].materialize()
        self._init_llr(paths[0].L[0], llr_ch)

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                path.materialize()
                L = path.L[0]
                B = path.B[0]
                self._update_llrs(L, B, l)
                llr_val = L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    path.pm += penalty
                    B[l, self.n] = 0
                    self._update_bits(B, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = path.fork()
                        child.materialize()
                        cL = child.L[0]
                        cB = child.B[0]
                        cL[:] = L
                        cB[:] = B
                        penalty = 0.0 if (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0) else abs(llr_val)
                        child.pm += penalty
                        cB[l, self.n] = bit
                        self._update_bits(cB, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                p.materialize()
                bits = p.B[0][:, self.n][self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        best.materialize()
        return best.B[0][:, self.n].astype(int), best.pm
