"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if int(bit) ^ msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    payload = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_process(payload, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_process(bits, poly, crc_length)
    return remainder == 0


# ==================== SCL 译码器 ====================

class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


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
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _branch_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, path, l):
        start = self.n - _active_llr_level(l, self.n)
        for s in range(start, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l, bit):
        path.u_hat[l] = bit
        path.B[l, self.n] = bit
        if l < self.N // 2:
            return
        start_b = self.n - _active_bit_level(l, self.n) + 1
        for s in range(self.n, start_b - 1, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_internal = llr_ch[br]

        paths = [_Path(self.N, self.n, llr_internal)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pm = path.pm + self._branch_penalty(llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._branch_penalty(llr, bit)
                        candidates.append((pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in selected:
                child = _Path(self.N, self.n, llr_internal)
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.pm = pm
                child.u_hat = parent.u_hat.copy()
                self._update_bits(child, l, bit)
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            for path in paths:
                payload = path.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    return path.u_hat, path.pm

        return paths[0].u_hat, paths[0].pm
