"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_operation,
    g_operation,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    expected = _crc_remainder(bits[:-crc_length], poly, crc_length)
    received = 0
    for i in range(crc_length):
        received = (received << 1) | int(bits[-crc_length + i])
    return expected == received


class _PathState:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：L/B 数组按需复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)
        self.brp = np.array([bit_reversed(i, self.n) for i in range(N)], dtype=int)

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.brp]
        paths = [_PathState(self.N, self.n, llr_ch)]

        for phase in range(self.N):
            l = bit_reversed(phase, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    new_path = _PathState(self.N, self.n, llr_ch)
                    new_path.pm = path.pm + penalty
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.u_hat = path.u_hat.copy()
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = _PathState(self.N, self.n, llr_ch)
                        new_path.pm = path.pm + self._pm_penalty(llr, bit)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_crc_pm = np.inf
        best_any = paths[0]
        best_any_pm = paths[0].pm

        for path in paths:
            if path.pm < best_any_pm:
                best_any = path
                best_any_pm = path.pm
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length) and path.pm < best_crc_pm:
                    best_crc = path
                    best_crc_pm = path.pm

        chosen = best_crc if best_crc is not None else best_any
        return chosen.u_hat.copy(), chosen.pm
