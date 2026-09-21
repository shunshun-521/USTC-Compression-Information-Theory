"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _lower_llr,
    f_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_update(reg, bit, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    msb_mask = 1 << (crc_length - 1)
    fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
    reg = (reg << 1) & mask
    if fb:
        reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, crc_length)
    return reg == 0


def _path_metric_update(pm, llr, bit):
    """路径度量更新：与 LLR 符号不一致时加 |LLR| 惩罚。"""
    hard = 0 if llr >= 0 else 1
    penalty = 0.0 if bit == hard else abs(llr)
    return pm + penalty


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组引用，分裂时复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = bit_reversal_permutation(N)

    def _update_llrs(self, L, B, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = self.decode_order[phi]
            new_paths = []

            for path in paths:
                self._update_llrs(path.L, path.B, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pm = _path_metric_update(path.pm, llr, 0)
                    new_path = _Path(self.N, self.n, llr_ch)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.pm = pm
                    new_path.u_hat = path.u_hat.copy()
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path.B, l)
                    new_paths.append(new_path)
                else:
                    for bit in (0, 1):
                        pm = _path_metric_update(path.pm, llr, bit)
                        new_path = _Path(self.N, self.n, llr_ch)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = pm
                        new_path.u_hat = path.u_hat.copy()
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path.B, l)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p.u_hat)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _crc_valid(self, u_hat):
        info_positions = np.where(~self.frozen_bits)[0]
        payload = u_hat[info_positions]
        return crc_check(payload, self.crc_length)
