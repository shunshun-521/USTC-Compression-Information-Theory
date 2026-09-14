"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _prepare_channel_llrs,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class PathState:
    """单条译码路径状态"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = _prepare_channel_llrs(llr_ch)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _copy_path(self, path):
        new_path = PathState(self.N, self.n, np.zeros(self.N))
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _crc_passes(self, u_hat):
        if self.crc_length == 0:
            return True
        info_bits = u_hat[self.info_indices]
        return crc_check(info_bits, self.crc_length)

    def decode(self, llr_ch):
        paths = [PathState(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)

            if l in self.frozen_set:
                for path in paths:
                    llr = path.L[l, self.n]
                    path.pm += self._pm_penalty(llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n, self.N)
                continue

            new_paths = []
            for path in paths:
                llr = path.L[l, self.n]
                for bit in (0, 1):
                    child = self._copy_path(path)
                    child.pm += self._pm_penalty(llr, bit)
                    child.u_hat[l] = bit
                    child.B[l, self.n] = bit
                    _update_bits(child.B, l, self.n, self.N)
                    new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        valid = [p for p in paths if self._crc_passes(p.u_hat)]
        best = min(valid if valid else paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
