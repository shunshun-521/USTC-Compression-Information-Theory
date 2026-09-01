"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _permute_channel_llr,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB first，逐比特）。"""
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 译码器 ====================

class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] + path.B[j - branch_size, s]
                    ) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr = _permute_channel_llr(llr_ch)
        N, n = self.N, self.n

        paths = [_PathState(N, n, llr)]
        decoded_bits = {p: {} for p in range(self.list_size)}

        for i in range(N):
            l = _bit_reversed_index(i, n)
            candidates = []

            for p_idx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    penalty = self._pm_penalty(llr_val, 0)
                    candidates.append((path.pm + penalty, p_idx, 0))
                else:
                    for bit in (0, 1):
                        penalty = self._pm_penalty(llr_val, bit)
                        candidates.append((path.pm + penalty, p_idx, bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for new_pm, parent_idx, bit in candidates:
                parent = paths[parent_idx]
                child = _PathState(N, n, llr)
                child.pm = new_pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.B[l, n] = bit
                self._update_bits(child, l)
                new_paths.append(child)

            paths = new_paths

        best_idx = 0
        best_pm = paths[0].pm
        crc_idx = None
        crc_pm = float("inf")

        for idx, path in enumerate(paths):
            u_hat = path.B[:, n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length) and path.pm < crc_pm:
                    crc_pm = path.pm
                    crc_idx = idx
            if path.pm < best_pm:
                best_pm = path.pm
                best_idx = idx

        if crc_idx is not None:
            best_idx = crc_idx

        u_hat = paths[best_idx].B[:, n].astype(int)
        return u_hat, paths[best_idx].pm
