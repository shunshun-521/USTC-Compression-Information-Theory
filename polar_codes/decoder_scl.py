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
    _lower_llr,
    f_operation,
    sc_decode,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, reg_bits):
    reg = 0
    mask = (1 << reg_bits) - 1
    for b in bits:
        reg ^= int(b) << (reg_bits - 1)
        if reg & (1 << (reg_bits - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly, r = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, r = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, r)
    crc_bits = np.array([(rem >> (r - 1 - i)) & 1 for i in range(r)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly, r = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, r = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, r)
    return rem == 0


def _path_metric_add(pm, llr, u_bit):
    """PM 更新：与 LLR 符号不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    if int(u_bit) != hard:
        pm += abs(float(llr))
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u_stage")

    def __init__(self, N, n, llr_perm):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_perm
        self.pm = 0.0
        self.u_stage = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_stage = self.u_stage.copy()
        return p

    def update_llrs(self, l, n):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            N = self.L.shape[0]
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(
                        self.L[j, s], self.L[j + branch_size, s]
                    )
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def update_bits(self, l, n):
        N = self.L.shape[0]
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


class SCLDecoder:
    """SCL 译码器（路径分裂时拷贝状态，实现简单 Lazy Copy 语义）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size <= 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_perm = llr_ch[self.br]
        paths = [_Path(self.N, self.n, llr_perm)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []
            for path in paths:
                path.update_llrs(l, self.n)
                llr_leaf = path.L[l, self.n]
                if l in self.frozen_set:
                    path.pm = _path_metric_add(path.pm, llr_leaf, 0)
                    path.B[l, self.n] = 0
                    path.u_stage[l] = 0
                    path.update_bits(l, self.n)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm = _path_metric_add(child.pm, llr_leaf, bit)
                        child.B[l, self.n] = bit
                        child.u_stage[l] = bit
                        child.update_bits(l, self.n)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_pm = paths[0].pm
        candidates = paths
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_nat = p.u_stage
                payload = u_nat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                candidates = valid
        best = min(candidates, key=lambda p: p.pm)
        return best.u_stage.astype(np.int8), float(best.pm)
