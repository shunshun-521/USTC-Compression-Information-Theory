"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
    precompute_sc_indices,
    _bit_reversed_index,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _branch_pm(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        penalty = 0.0 if bit == hard else abs(llr)
        return pm + penalty

    def _update_llrs(self, paths, l):
        n, N = self.n, self.N
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    for p in paths:
                        top = p.L[j, s]
                        btm = p.L[j + branch_size, s]
                        p.L[j, s + 1] = f_operation(top, btm)
                else:
                    for p in paths:
                        btm = p.L[j, s]
                        top = p.L[j - branch_size, s]
                        top_bit = p.B[j - branch_size, s + 1]
                        p.L[j, s + 1] = g_operation(top, btm, top_bit)

    def _update_bits(self, paths, l):
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    for p in paths:
                        p.B[j - branch_size, s - 1] = (
                            p.B[j, s] ^ p.B[j - branch_size, s]
                        )
                        p.B[j, s - 1] = p.B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat（自然序）, pm
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [_Path(N, n) for _ in range(self.list_size)]
        paths[0].L[:, 0] = llr_ch
        active = 1
        u_paths = [np.zeros(N, dtype=int) for _ in range(self.list_size)]

        for l in self.decode_order:
            self._update_llrs(paths[:active], l)

            candidates = []
            if self.frozen_bits[l]:
                for idx in range(active):
                    pm = self._branch_pm(paths[idx].pm, paths[idx].L[l, n], 0)
                    candidates.append((pm, idx, 0))
            else:
                for idx in range(active):
                    for bit in (0, 1):
                        pm = self._branch_pm(
                            paths[idx].pm, paths[idx].L[l, n], bit
                        )
                        candidates.append((pm, idx, bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            new_u = []
            for pm, old_idx, bit in candidates:
                p = _Path(N, n)
                p.pm = pm
                p.L = paths[old_idx].L.copy()
                p.B = paths[old_idx].B.copy()
                p.B[l, n] = bit
                new_paths.append(p)
                u = u_paths[old_idx].copy()
                u[l] = bit
                new_u.append(u)

            paths = new_paths
            u_paths = new_u
            active = len(candidates)

            self._update_bits(paths, l)

        best_idx = 0
        best_pm = paths[0].pm
        if self.crc_length > 0:
            crc_pass = []
            for idx in range(active):
                info_bits = self._extract_info_bits(u_paths[idx])
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append((paths[idx].pm, idx))
            if crc_pass:
                crc_pass.sort(key=lambda x: x[0])
                best_pm, best_idx = crc_pass[0]
            else:
                for idx in range(1, active):
                    if paths[idx].pm < best_pm:
                        best_pm = paths[idx].pm
                        best_idx = idx
        else:
            for idx in range(1, active):
                if paths[idx].pm < best_pm:
                    best_pm = paths[idx].pm
                    best_idx = idx

        return u_paths[best_idx], best_pm

    def _extract_info_bits(self, u_hat):
        info_mask = ~self.frozen_bits
        return u_hat[info_mask]
