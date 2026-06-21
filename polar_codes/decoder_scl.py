"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from encoder import inverse_bit_reversal_permutation
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    f_operation,
    g_operation,
    path_metric_penalty,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_register(bits, crc_length):
    """CRC 寄存器更新（MSB 优先）"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    reg = _crc_register(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    received = bits[-crc_length:]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(received, expected)


class _Path:
    __slots__ = ("pm", "L", "C", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        inv_br = inverse_bit_reversal_permutation(N)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.C = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch[inv_br]
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        child = _Path.__new__(_Path)
        child.pm = self.pm
        child.L = self.L.copy()
        child.C = self.C.copy()
        child.u_hat = self.u_hat.copy()
        return child


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制路径状态）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - active_llr_level(l, self.n), self.n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        top_bit = path.C[j - branch_size, s + 1]
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s], path.L[j, s], top_bit
                        )

    def _propagate_bits(self, paths, l):
        if l < self.N // 2:
            return
        for path in paths:
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.C[j - branch_size, s - 1] = (
                            path.C[j, s] + path.C[j - branch_size, s]
                        ) % 2
                        path.C[j, s - 1] = path.C[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = bit_reversed_index(phi, self.n)
            self._update_llrs(paths, l)
            llr_val = paths[0].L[l, self.n]

            if l in self.frozen_set:
                for path in paths:
                    path.u_hat[l] = 0
                    path.C[l, self.n] = 0
                    path.pm += path_metric_penalty(llr_val, 0)
                self._propagate_bits(paths, l)
                continue

            new_paths = []
            for path in paths:
                cur_llr = path.L[l, self.n]
                for bit in (0, 1):
                    child = path.copy()
                    child.u_hat[l] = bit
                    child.C[l, self.n] = bit
                    child.pm += path_metric_penalty(cur_llr, bit)
                    new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]
            self._propagate_bits(paths, l)

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat.copy(), path.pm

        best = paths[0]
        return best.u_hat.copy(), best.pm
