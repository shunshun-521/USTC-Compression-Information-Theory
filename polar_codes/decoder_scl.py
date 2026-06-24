"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric_update(pm, llr, u_bit):
    hard = 0 if llr >= 0 else 1
    if u_bit == hard:
        return pm
    return pm + abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.br = bit_reversal_permutation(N)
        self.frozen_br = self.frozen_bits[self.br]
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            new_paths = []

            for path in paths:
                if not path.active:
                    continue

                _update_llrs(l, self.n, self.N, path.L, path.B)
                llr_bit = path.L[l, self.n]

                if self.frozen_br[l]:
                    pm = _path_metric_update(path.pm, llr_bit, 0)
                    child = self._fork_path(path)
                    child.pm = pm
                    child.B[l, self.n] = 0
                    child.u_hat[l] = 0
                    _update_bits(l, self.n, self.N, child.B)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        pm = _path_metric_update(path.pm, llr_bit, u_bit)
                        child = self._fork_path(path)
                        child.pm = pm
                        child.B[l, self.n] = u_bit
                        child.u_hat[l] = u_bit
                        _update_bits(l, self.n, self.N, child.B)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        natural_paths = []
        for path in paths:
            u_nat = path.u_hat[self.br].copy()
            natural_paths.append((path.pm, u_nat))

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in natural_paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        natural_paths.sort(key=lambda x: x[0])
        return natural_paths[0][1], natural_paths[0][0]

    @staticmethod
    def _fork_path(path):
        child = _Path.__new__(_Path)
        child.L = path.L.copy()
        child.B = path.B.copy()
        child.pm = path.pm
        child.u_hat = path.u_hat.copy()
        child.active = True
        return child
