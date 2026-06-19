"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _bit_reversed_index,
    _update_bits,
    _update_llrs,
    precompute_sc_indices,
    reorder_channel_llr,
    sc_decode,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """
    SCL 译码器。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        _, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = reorder_channel_llr(llr_ch)
        paths = [_Path(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                cur_llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    child = _Path(self.n, self.N)
                    child.L[:] = path.L
                    child.B[:] = path.B
                    child.u_hat[:] = path.u_hat
                    child.pm = path.pm + _pm_penalty(cur_llr, 0)
                    child.u_hat[l] = 0
                    _update_bits(child.B, l, self.n, self.N)
                    candidates.append(child)
                else:
                    for u_val in (0, 1):
                        child = _Path(self.n, self.N)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.u_hat[:] = path.u_hat
                        child.pm = path.pm + _pm_penalty(cur_llr, u_val)
                        child.u_hat[l] = u_val
                        _update_bits(child.B, l, self.n, self.N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            chosen = min(valid, key=lambda p: p.pm) if valid else min(
                paths, key=lambda p: p.pm
            )
        else:
            chosen = min(paths, key=lambda p: p.pm)

        return chosen.u_hat.copy(), chosen.pm
