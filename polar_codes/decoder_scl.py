"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _channel_llr_to_decoder,
    _xor_combine,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class PathState:
    __slots__ = ("llr", "u_hat", "pm", "node_partials")

    def __init__(self, N):
        self.llr = None
        self.u_hat = np.zeros(N, dtype=int)
        self.pm = 0.0
        self.node_partials = {}


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 1 if llr < 0 else 0
        return 0.0 if bit == hard else abs(llr)

    def _extend_paths(self, paths, depth, node, y):
        if depth == self.n - 1:
            new_paths = []
            for path in paths:
                for bit in ([0] if self.frozen_bits[node] else [0, 1]):
                    if self.frozen_bits[node]:
                        bit = 0
                    child = PathState(self.N)
                    child.llr = path.llr
                    child.u_hat = path.u_hat.copy()
                    child.pm = path.pm + self._pm_penalty(y[0], bit)
                    child.u_hat[node] = bit
                    child.node_partials = dict(path.node_partials)
                    child.node_partials[node] = np.array([bit], dtype=int)
                    new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            return new_paths[: self.list_size]

        half = len(y) // 2
        left_llr = f_operation(y[:half], y[half:])
        left_paths = self._extend_paths(paths, depth + 1, 2 * node, left_llr)

        expanded = []
        for path in left_paths:
            left_partial = path.node_partials.get(2 * node, np.array([0], dtype=int))
            right_llr = g_operation(y[:half], y[half:], left_partial)
            right_paths = self._extend_paths([path], depth + 1, 2 * node + 1, right_llr)
            for rp in right_paths:
                right_partial = rp.node_partials.get(2 * node + 1, np.array([0], dtype=int))
                combined = _xor_combine(left_partial, right_partial)
                rp.node_partials[node] = combined
                expanded.append(rp)

        expanded.sort(key=lambda p: p.pm)
        return expanded[: self.list_size]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr = _channel_llr_to_decoder(llr_ch)
        init = PathState(self.N)
        init.llr = llr
        paths = self._extend_paths([init], 0, 0, llr)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
