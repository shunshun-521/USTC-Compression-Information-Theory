"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr_boxplus,
    _lower_llr_boxplus,
    _get_sc_cache,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """按位模 2 除法计算 CRC 余数"""
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


def _llr_penalty(llr, u):
    """路径度量惩罚：判决与 LLR 符号不一致时加 |LLR|"""
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )
        self.rev = bit_reversal_permutation(N)
        _, self.llr_layer_vec, self.bit_layer_vec = _get_sc_cache(N)

    def _init_paths(self, llr):
        paths = [{
            'L': np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            'B': np.full((self.N, self.n + 1), np.nan),
            'pm': 0.0,
            'u': np.zeros(self.N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr
        return paths

    def _update_llrs(self, path, phi):
        l = _bit_reversed(phi, self.n)
        for s in self.llr_layer_vec[phi]:
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            L, B = path['L'], path['B']
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr_boxplus(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )
        return L[l, self.n]

    def _update_bits(self, path, phi, u_val):
        l = _bit_reversed(phi, self.n)
        B = path['B']
        B[l, self.n] = u_val
        path['u'][l] = u_val
        for s in self.bit_layer_vec[phi]:
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        paths = self._init_paths(llr)

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                cur_llr = self._update_llrs(path, phi)

                if l in self.frozen_set:
                    p = self._lazy_copy(path)
                    p['pm'] += _llr_penalty(cur_llr, 0)
                    self._update_bits(p, phi, 0)
                    new_paths.append(p)
                else:
                    for u_val in (0, 1):
                        p = self._lazy_copy(path)
                        p['pm'] += _llr_penalty(cur_llr, u_val)
                        self._update_bits(p, phi, u_val)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        return self._select_best(paths)

    def _lazy_copy(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u': path['u'].copy(),
        }

    def _select_best(self, paths):
        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p['u'][self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p['pm'])
                return best['u'], best['pm']
        best = min(paths, key=lambda p: p['pm'])
        return best['u'], best['pm']
