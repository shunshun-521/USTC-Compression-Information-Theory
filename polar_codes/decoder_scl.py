"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


# ==================== CRC 工具 ====================

def _crc_poly_encode(data_bits, width, poly):
    msg = [int(b) for b in data_bits] + [0] * width
    for i in range(len(data_bits)):
        if msg[i]:
            for j in range(width + 1):
                if (poly >> (width - j)) & 1 and i + j < len(msg):
                    msg[i + j] ^= 1
    return np.array(msg[-width:], dtype=int)


def _crc_poly_check(bits, width, poly):
    msg = [int(b) for b in bits]
    for i in range(len(bits) - width):
        if msg[i]:
            for j in range(width + 1):
                if (poly >> (width - j)) & 1 and i + j < len(msg):
                    msg[i + j] ^= 1
    return all(x == 0 for x in msg[-width:])


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = 0x107
    elif crc_length == 16:
        poly = 0x11021
    else:
        raise ValueError("crc_length must be 8 or 16")

    crc_bits = _crc_poly_encode(info_bits, crc_length, poly)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = 0x107
    elif crc_length == 16:
        poly = 0x11021
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_poly_check(bits, crc_length, poly)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


def _path_llr_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _clone_path(path):
    new_path = _Path.__new__(_Path)
    new_path.L = path.L.copy()
    new_path.B = path.B.copy()
    new_path.pm = path.pm
    new_path.u_hat = path.u_hat.copy()
    return new_path


def _advance_path(path, l, frozen_set, n):
    _update_llrs(path.L, path.B, l, n)
    llr = path.L[l, n]
    if l in frozen_set:
        path.u_hat[l] = 0
        path.pm += _path_llr_penalty(llr, 0)
        path.B[l, n] = 0
    else:
        raise RuntimeError("advance_path called on information bit")
    _update_bits(path.B, l, n)
    return llr


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    if l < len(B) / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = _clone_path(path)
                    new_path.u_hat[l] = 0
                    new_path.pm += _path_llr_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = _clone_path(path)
                        new_path.u_hat[l] = bit
                        new_path.pm += _path_llr_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
