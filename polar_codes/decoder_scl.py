"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    lower_llr_exact,
    upper_llr_exact,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(recomputed, bits)


class _Path:
    """SCL 单条路径（Lazy Copy）。"""

    __slots__ = ("L", "B", "pm", "parent_L", "parent_B", "copy_L", "copy_B")

    def __init__(self, N, n):
        self.L = None
        self.B = None
        self.pm = 0.0
        self.parent_L = None
        self.parent_B = None
        self.copy_L = False
        self.copy_B = False

    def ensure_L(self, template):
        if self.L is None:
            self.L = template.copy()
        elif self.copy_L and self.parent_L is not None:
            self.L = self.parent_L.L.copy()
            self.copy_L = False

    def ensure_B(self, template):
        if self.B is None:
            self.B = template.copy()
        elif self.copy_B and self.parent_B is not None:
            self.B = self.parent_B.B.copy()
            self.copy_B = False


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs_all_paths(self, paths, l):
        for path in paths:
            path.ensure_L(paths[0].L)
            L = path.L
            for s in range(self.n - active_llr_level(l, self.n), self.n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = upper_llr_exact(L[j, s], L[j + branch_size, s])
                    else:
                        top_bit = path.B[j - branch_size, s + 1]
                        L[j, s + 1] = lower_llr_exact(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits_path(self, path, l):
        path.ensure_B(path.B if path.B is not None else np.zeros((self.N, self.n + 1), int))
        B = path.B
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        base_L = np.full((N, n + 1), np.nan, dtype=np.float64)
        base_L[:, 0] = llr_ch
        base_B = np.zeros((N, n + 1), dtype=int)

        root = _Path(N, n)
        root.L = base_L.copy()
        root.B = base_B.copy()
        paths = [root]

        for phi in range(N):
            l = int(f"{phi:0{n}b}"[::-1], 2)
            self._update_llrs_all_paths(paths, l)

            new_paths = []
            for path in paths:
                llr_bit = path.L[l, n]
                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(llr_bit, 0)
                    path.pm += penalty
                    path.B[l, n] = 0
                    self._update_bits_path(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(N, n)
                        child.parent_L = path
                        child.parent_B = path
                        child.copy_L = True
                        child.copy_B = True
                        child.pm = path.pm + self._path_metric_penalty(llr_bit, bit)
                        child.ensure_L(path.L)
                        child.ensure_B(path.B)
                        child.L = child.L.copy()
                        child.B = child.B.copy()
                        child.copy_L = False
                        child.copy_B = False
                        child.B[l, n] = bit
                        self._update_bits_path(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        u_hat_all = np.array([p.B[:, n].astype(int) for p in paths])
        pms = np.array([p.pm for p in paths])

        if self.crc_length > 0:
            valid = []
            for idx, u in enumerate(u_hat_all):
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(idx)
            if valid:
                best = valid[np.argmin(pms[valid])]
            else:
                best = int(np.argmin(pms))
        else:
            best = int(np.argmin(pms))

        return u_hat_all[best], float(pms[best])
