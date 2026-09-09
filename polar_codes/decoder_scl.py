"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation, bit_reversed_index
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _frozen_set_from_mask,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class Path:
    """SCL 单条路径状态。"""

    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_set_from_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, self.N, block):
                if j % block < branch:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch, s], path.L[j, s], path.B[j - branch, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    path.B[j - branch, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _path_llr(self, path, l):
        return path.L[l, self.n]

    def _branch_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [Path(self.N, self.n, llr.copy())]
        decode_order = [bit_reversed_index(i, self.n) for i in range(self.N)]

        for l in decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_val = self._path_llr(path, l)

                if l in self.frozen_set:
                    pen = self._branch_penalty(llr_val, 0)
                    path.pm += pen
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = Path(self.N, self.n, path.L[:, 0].copy())
                        child.L[:, 1:] = np.array(path.L[:, 1:], copy=True)
                        child.B[:, 1:] = np.array(path.B[:, 1:], copy=True)
                        child.pm = path.pm + self._branch_penalty(llr_val, u_bit)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = u_bit
                        child.B[l, self.n] = u_bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm


def scl_equivalent_sc(llr_ch, frozen_bits):
    """L=1 的 SCL 应与 SC 一致。"""
    N = len(llr_ch)
    dec = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_scl, _ = dec.decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
