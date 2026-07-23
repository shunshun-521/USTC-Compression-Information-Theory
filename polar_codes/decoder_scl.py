"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Vangala Permuted SCD 结构
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _frozen_mask_to_set,
)
from encoder import bit_reversed_index

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << crc_length
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << crc_length
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class _SCLPath:
    __slots__ = ("pm", "L", "B", "parent", "_L_ref", "_B_ref")

    def __init__(self, N, n, parent=None):
        self.pm = 0.0
        self.parent = parent
        if parent is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self._L_ref = None
            self._B_ref = None
        else:
            self.L = None
            self.B = None
            self._L_ref = parent
            self._B_ref = parent

    def get_L(self):
        return self.L if self.L is not None else self._L_ref.L

    def get_B(self):
        return self.B if self.B is not None else self._B_ref.B

    def materialize(self):
        if self.L is None:
            self.L = self._L_ref.get_L().copy()
            self._L_ref = None
        if self.B is None:
            self.B = self._B_ref.get_B().copy()
            self._B_ref = None


def _update_llrs(path, l, n, N):
    L = path.get_L()
    B = path.get_B()
    for s in range(n - _active_llr_level(l, n), n):
        block = 1 << (s + 1)
        half = block // 2
        for j in range(l, N, block):
            if j % block < half:
                L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
            else:
                L[j, s + 1] = g_operation(L[j - half, s], L[j, s], B[j - half, s + 1])


def _update_bits(path, l, n, N):
    if l < N // 2:
        return
    B = path.get_B()
    for s in range(n, n - _active_bit_level(l, n), -1):
        block = 1 << s
        half = block // 2
        for j in range(l, -1, -block):
            if j % block >= half:
                B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                B[j, s - 1] = B[j, s]


def _metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _frozen_mask_to_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = sorted(set(range(N)) - self.frozen_set)
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        root = _SCLPath(N, n)
        root.get_L()[:, 0] = llr_ch
        paths = [root]

        for l in self.decode_order:
            for path in paths:
                path.materialize()
                _update_llrs(path, l, n, N)

            new_paths = []
            if l in self.frozen_set:
                for path in paths:
                    path.materialize()
                    llr = path.get_L()[l, n]
                    path.pm += _metric_penalty(llr, 0)
                    path.get_B()[l, n] = 0
                    _update_bits(path, l, n, N)
                    new_paths.append(path)
            else:
                for path in paths:
                    llr = path.get_L()[l, n]
                    for bit in (0, 1):
                        child = _SCLPath(N, n, parent=path)
                        child.pm = path.pm + _metric_penalty(llr, bit)
                        child.materialize()
                        child.get_B()[l, n] = bit
                        _update_bits(child, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            u_hat = path.get_B()[:, n].astype(int)
            candidates.append((path.pm, u_hat))

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                candidates = valid

        pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat, pm
