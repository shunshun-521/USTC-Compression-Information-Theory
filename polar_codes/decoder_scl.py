"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _cn_op,
    _prepare_llr,
    _vn_op,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(info_bits, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    return np.concatenate([info_bits, _crc_remainder(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    return np.array_equal(bits[-crc_length:], _crc_remainder(info, crc_length))


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（PSC 非递归 + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(np.asarray(frozen_bits, dtype=int) == 0)[0]

    def _penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _cn_op(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = _vn_op(
                        path.L[j - branch_size, s], path.L[j, s], int(top_bit)
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _clone(self, path):
        p = _Path(self.N, self.n, path.L[:, 0])
        p.pm = path.pm
        p.L = path.L.copy()
        p.B = path.B.copy()
        p.u_hat = path.u_hat.copy()
        return p

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, np.array([1 if i in self.frozen_set else 0 for i in range(self.N)])), 0.0

        llr = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr)]
        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_l = path.L[l, self.n]
                if l in self.frozen_set:
                    p = self._clone(path)
                    p.pm += self._penalty(llr_l, 0)
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    self._update_bits(p, l)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = self._clone(path)
                        p.pm += self._penalty(llr_l, u)
                        p.u_hat[l] = u
                        p.B[l, self.n] = u
                        self._update_bits(p, l)
                        candidates.append(p)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
