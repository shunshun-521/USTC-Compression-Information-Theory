"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_boxplus,
    g_operation,
    reorder_channel_llr,
    active_llr_level,
    active_bit_level,
    precompute_sc_indices,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
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

    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    """SCL 单条路径（Lazy Copy）。"""

    __slots__ = ("L", "B", "pm", "parent", "copied_L", "copied_B")

    def __init__(self, N, n, llr_ch, parent=None):
        self.parent = parent
        self.copied_L = False
        self.copied_B = False
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.full((N, n + 1), np.nan)
            self.L[:, 0] = llr_ch
            self.pm = 0.0
        else:
            self.L = parent.L
            self.B = parent.B
            self.pm = parent.pm

    def ensure_writable(self):
        if not self.copied_L:
            self.L = self.L.copy()
            self.copied_L = True
        if not self.copied_B:
            self.B = self.B.copy()
            self.copied_B = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, _ = precompute_sc_indices(N)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = reorder_channel_llr(np.asarray(llr_ch, dtype=float))
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                path.ensure_writable()
                self._update_llrs(path.L, path.B, l)

                llr_bit = path.L[l, self.n]
                is_frozen = l in self.frozen_set

                if is_frozen:
                    penalty = abs(llr_bit) if llr_bit < 0 else 0.0
                    path.pm += penalty
                    path.B[l, self.n] = 0
                    self._update_bits(path.B, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, llr_ch, parent=path)
                        child.ensure_writable()
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.copied_L = True
                        child.copied_B = True
                        child.pm = path.pm
                        if (bit == 0 and llr_bit < 0) or (bit == 1 and llr_bit >= 0):
                            child.pm += abs(llr_bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child.B, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        u_hat, pm = self._select_best_path(paths)
        return u_hat, pm

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u = self._extract_u(p.B)
                info_bits = u[self.info_indices]
                if len(info_bits) >= self.crc_length and crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p.pm)
        return self._extract_u(best.B), best.pm

    def _extract_u(self, B):
        u_hat = np.zeros(self.N, dtype=int)
        for l in self.decode_order:
            val = B[l, self.n]
            u_hat[l] = 0 if np.isnan(val) else int(val)
        return u_hat

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    bj = 0 if np.isnan(B[j, s]) else int(B[j, s])
                    btop = 0 if np.isnan(B[j - branch_size, s]) else int(B[j - branch_size, s])
                    B[j - branch_size, s - 1] = bj ^ btop
                    B[j, s - 1] = bj
