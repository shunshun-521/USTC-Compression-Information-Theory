"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    sc_decode,
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length, reflected=True):
    if reflected:
        reg ^= int(bit)
        for _ in range(crc_length):
            if reg & 1:
                reg = (reg >> 1) ^ poly
            else:
                reg >>= 1
        return reg & ((1 << crc_length) - 1)

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
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_step(reg, bit, poly, crc_length, reflected=True)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg = _crc_step(reg, bit, poly, crc_length, reflected=True)
    return reg == 0


class _SCLPath:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """
    SCL 译码器。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.br = bit_reversal_permutation(N)

        if list_size == 1 and crc_length == 0:
            self._use_sc = True
        else:
            self._use_sc = False

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self._use_sc:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = llr_ch[self.br]

        paths = [_SCLPath(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = abs(llr) if llr < 0 else 0.0
                    candidates.append((path.pm + penalty, path, 0))
                else:
                    u0_penalty = 0.0 if llr >= 0 else abs(llr)
                    u1_penalty = abs(llr) if llr >= 0 else 0.0
                    candidates.append((path.pm + u0_penalty, path, 0))
                    candidates.append((path.pm + u1_penalty, path, 1))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in selected:
                child = _SCLPath(self.N, self.n, llr_ch)
                child.pm = pm
                child.L[:, :] = parent.L
                child.B[:, :] = parent.B
                child.B[l, self.n] = bit
                self._update_bits(child, l)
                new_paths.append(child)

            paths = new_paths

        best_path = None
        best_pm = float("inf")

        if self.crc_length > 0:
            for path in paths:
                u_hat = path.B[:, self.n].astype(int)
                info_bits = u_hat[self.frozen_bits == 0]
                if crc_check(info_bits, self.crc_length):
                    if path.pm < best_pm:
                        best_pm = path.pm
                        best_path = u_hat

        if best_path is None:
            for path in paths:
                if path.pm < best_pm:
                    best_pm = path.pm
                    best_path = path.B[:, self.n].astype(int)

        return best_path, best_pm
