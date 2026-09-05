"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _hard_decision,
    _update_bits,
    _update_llrs,
    bit_reversed_index,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) | int(bit)
        if msb ^ int(bit):
            reg ^= poly
    for _ in range(crc_length):
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length <= 0 or len(bits) < crc_length:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, u_bit):
        hard = _hard_decision(llr)
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [PathState(N, n)]
        paths[0].L[:, 0] = llr_ch

        for phi_natural in range(N):
            l = bit_reversed_index(phi_natural, n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr = path.L[l, n]

                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(llr, 0)
                    child = PathState(N, n)
                    child.L[:] = path.L
                    child.B[:] = path.B
                    child.pm = path.pm + penalty
                    child.u_hat[:] = path.u_hat
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    _update_bits(child.B, l, n, N)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = PathState(N, n)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.pm = path.pm + self._path_metric_penalty(llr, u_bit)
                        child.u_hat[:] = path.u_hat
                        child.u_hat[l] = u_bit
                        child.B[l, n] = u_bit
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_ok = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(crc_ok if crc_ok else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
