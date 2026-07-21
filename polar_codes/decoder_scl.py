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
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


# CRC 多项式
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数"""
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


def _update_llrs_path(L, B, l, n):
    """单路径 LLR 更新"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits_path(B, l, n):
    """单路径比特回传"""
    if l < B.shape[0] / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2**s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = np.zeros((self.list_size, self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.list_size, self.N, self.n + 1), dtype=int)
        pm = np.zeros(self.list_size, dtype=np.float64)
        active = 1

        for p in range(self.list_size):
            L[p, :, 0] = llr_ch[self.br]

        for phase_idx, l in enumerate(self.decode_order):
            new_paths = []

            for p in range(active):
                _update_llrs_path(L[p], B[p], l, self.n)
                cur_llr = L[p, l, self.n]

                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(cur_llr, 0)
                    B_new = B[p].copy()
                    L_new = L[p].copy()
                    B_new[l, self.n] = 0
                    _update_bits_path(B_new, l, self.n)
                    new_paths.append((pm[p] + penalty, L_new, B_new))
                else:
                    for bit in (0, 1):
                        penalty = self._path_metric_penalty(cur_llr, bit)
                        B_new = B[p].copy()
                        L_new = L[p].copy()
                        B_new[l, self.n] = bit
                        _update_bits_path(B_new, l, self.n)
                        new_paths.append((pm[p] + penalty, L_new, B_new))

            new_paths.sort(key=lambda x: x[0])
            active = min(self.list_size, len(new_paths))
            for p in range(active):
                pm[p], L[p], B[p] = new_paths[p]

        candidates = []
        for p in range(active):
            u_hat = B[p, :, self.n].astype(int)
            candidates.append((pm[p], u_hat))

        if self.crc_length > 0:
            crc_pass = [(pm_, u) for pm_, u in candidates if crc_check(u, self.crc_length)]
            if crc_pass:
                crc_pass.sort(key=lambda x: x[0])
                return crc_pass[0][1], crc_pass[0][0]

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][0]
