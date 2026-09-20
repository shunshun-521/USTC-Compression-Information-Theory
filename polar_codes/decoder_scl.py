"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    frozen_mask_to_bool,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07), r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class Path:
    """SCL 单条路径（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "parent", "branch_bit")

    def __init__(self, N, n, llr_ch, parent=None, branch_bit=None):
        self.parent = parent
        self.branch_bit = branch_bit
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.full((N, n + 1), np.nan)
            self.L[:, 0] = llr_ch
            self.pm = 0.0
        else:
            self.L = parent.L
            self.B = parent.B
            self.pm = parent.pm

    def clone(self, branch_bit):
        return Path(None, None, None, parent=self, branch_bit=branch_bit)

    def copy_layers_for_write(self, layers_L, layers_B):
        if self.parent is not None:
            if not isinstance(self.L, np.ndarray) or self.L is self.parent.L:
                self.L = self.parent.L.copy()
            if not isinstance(self.B, np.ndarray) or self.B is self.parent.B:
                self.B = self.parent.B.copy()
            self.parent = None
        for s in layers_L:
            self.L[:, s] = self.L[:, s].copy()
        for s in layers_B:
            self.B[:, s] = self.B[:, s].copy()


def _update_llrs_path(path, l, n, write_layers):
    L, B = path.L, path.B
    for s in range(n - _active_llr_level(l, n), n):
        write_layers.add(s + 1)
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s],
                    L[j, s],
                    B[j - branch_size, s + 1],
                )


def _update_bits_path(path, l, n, N, write_layers):
    if l < N // 2:
        return
    B = path.B
    for s in range(n, n - _active_bit_level(l, n), -1):
        write_layers.add(s - 1)
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = int(B[j, s])


def _path_metric_update(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    if bit != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = frozen_mask_to_bool(frozen_bits)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_in = llr_ch[br]

        paths = [Path(self.N, self.n, llr_in)]
        decoded_bits = {}

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                layers_L = set()
                layers_B = set()
                path.copy_layers_for_write(layers_L, layers_B)
                _update_llrs_path(path, l, self.n, layers_L)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pm = _path_metric_update(path.pm, llr, 0)
                    path.pm = pm
                    path.B[l, self.n] = 0
                    _update_bits_path(path, l, self.n, self.N, layers_B)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = path.clone(bit)
                        child.copy_layers_for_write(set(), set())
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = _path_metric_update(path.pm, llr, bit)
                        child.B[l, self.n] = bit
                        _update_bits_path(child, l, self.n, self.N, set())
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            decoded_bits[l] = paths[0].B[l, self.n]

        best_path = paths[0]
        u_hat = best_path.B[:, self.n].astype(int)

        if self.crc_length > 0:
            info_bits = u_hat[self.info_indices]
            valid = []
            for path in paths:
                bits = path.B[:, self.n].astype(int)
                payload = bits[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            if valid:
                best_path = min(valid, key=lambda p: p.pm)
                u_hat = best_path.B[:, self.n].astype(int)

        return u_hat, best_path.pm
