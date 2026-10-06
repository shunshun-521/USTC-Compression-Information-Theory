"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import (
    _SCDCore,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    f_operation,
    g_operation,
    precompute_sc_indices,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, reg_bits):
    reg = 0
    for b in bits:
        reg ^= int(b) << (reg_bits - 1)
        for _ in range(reg_bits):
            if reg & (1 << (reg_bits - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << reg_bits) - 1)
            else:
                reg = (reg << 1) & ((1 << reg_bits) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


class PathState:
    """单条 SCL 路径（复用 SC 的 L/B 矩阵）"""

    __slots__ = ("pm", "core", "u_hat")

    def __init__(self, N, n, frozen_set, llr_ch):
        self.pm = 0.0
        self.core = _SCDCore(N, frozen_set)
        self.core.set_llr(llr_ch.copy())
        self.u_hat = np.zeros(N, dtype=int)

    def clone(self):
        p = PathState.__new__(PathState)
        p.pm = self.pm
        p.core = copy.deepcopy(self.core)
        p.u_hat = self.u_hat.copy()
        return p

    def process_bit(self, l, is_frozen):
        self.core._update_llrs(l)
        llr = self.core.L[l, self.core.n]
        if is_frozen:
            u = 0
            self.pm += _pm_penalty(llr, u)
        else:
            return llr  # 需要分裂
        self.core.B[l, self.core.n] = u
        self.u_hat[l] = u
        self.core._update_bits(l)
        return None

    def apply_bit(self, l, u, llr):
        self.pm += _pm_penalty(llr, u)
        self.core.B[l, self.core.n] = u
        self.u_hat[l] = u
        self.core._update_bits(l)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n, self.frozen_set, llr_ch)]

        for l in self.decode_order:
            is_frozen = l in self.frozen_set
            new_paths = []
            for path in paths:
                llr = path.process_bit(l, is_frozen)
                if is_frozen:
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p2 = path.clone()
                        p2.apply_bit(l, u, llr)
                        new_paths.append(p2)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
