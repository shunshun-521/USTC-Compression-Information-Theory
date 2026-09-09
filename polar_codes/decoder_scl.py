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
    _frozen_to_set,
    f_operation,
    g_operation,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_polynomial(crc_length):
    return CRC_POLYS[crc_length] | (1 << crc_length)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_polynomial(crc_length)
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    n = len(info_bits)
    for i in range(n):
        if msg[i]:
            for j in range(crc_length + 1):
                if (poly >> j) & 1:
                    msg[i + j] ^= 1
    return np.concatenate([info_bits, msg[n:].astype(np.int8)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_polynomial(crc_length)
    msg = bits.copy()
    n = len(bits) - crc_length
    for i in range(n):
        if msg[i]:
            for j in range(crc_length + 1):
                if (poly >> j) & 1:
                    msg[i + j] ^= 1
    return not np.any(msg[-crc_length:])


class Path:
    """SCL 单条路径。"""

    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr

    def copy(self):
        new = Path.__new__(Path)
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        new.L = self.L.copy()
        new.B = self.B.copy()
        return new


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_to_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top = path.L[j - branch_size, s]
                    btm = path.L[j, s]
                    bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(top, btm, bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
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

    def _path_metric_penalty(self, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return abs(llr_val) if u != hard else 0.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen:
                    new_path = path.copy()
                    new_path.pm += self._path_metric_penalty(llr_val, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._path_metric_penalty(llr_val, u)
                        new_path.u_hat[l] = u
                        new_path.B[l, self.n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
