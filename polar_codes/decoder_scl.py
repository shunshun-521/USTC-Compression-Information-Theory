"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _reorder_channel_llrs,
    f_operation,
    g_operation,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, poly, crc_length):
    """对 bits（已含尾部零填充）计算 CRC 余数。"""
    reg = 0
    top = 1 << crc_length
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & top:
            reg ^= poly
    return reg & (top - 1)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(padded, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC 约束。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    return _crc_remainder(bits, poly, crc_length) == 0


class _PathState:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _path_metric_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def _compute_llr_for_path(self, path, idx, l):
        for s in self.llr_layer_vec[idx]:
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, self.N, block):
                if j % block < half:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - half, s], path.L[j, s], path.B[j - half, s + 1]
                    )

    def _update_bits_for_path(self, path, idx, l):
        for s in self.bit_layer_vec[idx]:
            block = 1 << s
            half = block >> 1
            for j in range(l, -1, -block):
                if j % block >= half:
                    path.B[j - half, s - 1] = (path.B[j, s] + path.B[j - half, s]) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = _reorder_channel_llrs(np.asarray(llr_ch, dtype=np.float64))
        paths = [_PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for idx, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                self._compute_llr_for_path(path, idx, l)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._clone_path(path)
                    new_path.pm += self._path_metric_penalty(llr_val, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits_for_path(new_path, idx, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._path_metric_penalty(llr_val, u_bit)
                        new_path.B[l, self.n] = u_bit
                        new_path.u_hat[l] = u_bit
                        self._update_bits_for_path(new_path, idx, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            crc_pass = [
                p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            best = min(crc_pass or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm

    def _clone_path(self, path):
        new_path = _PathState(self.N, self.n)
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.u_hat = path.u_hat.copy()
        return new_path
