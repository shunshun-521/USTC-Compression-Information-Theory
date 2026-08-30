"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _prepare_channel_llr,
    _upper_llr,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
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
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        N, n, L = self.N, self.n, self.list_size
        paths = [_Path(N, n, llr_ch)]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            for path in paths:
                self._update_llrs(path, l)

            if self.frozen_bits[l]:
                for path in paths:
                    path.pm += self._pm_penalty(path.L[l, n], 0)
                    path.u_hat[l] = 0
                    path.B[l, n] = 0
                    self._update_bits(path, l)
            else:
                candidates = []
                for pi, path in enumerate(paths):
                    llr = path.L[l, n]
                    for bit in (0, 1):
                        candidates.append(
                            (path.pm + self._pm_penalty(llr, bit), pi, bit)
                        )
                candidates.sort(key=lambda x: x[0])

                new_paths = []
                seen = set()
                for pm, parent_idx, bit in candidates:
                    key = (parent_idx, bit)
                    if key in seen:
                        continue
                    seen.add(key)
                    if len(new_paths) >= L:
                        break
                    parent = paths[parent_idx]
                    child = _Path(N, n, llr_ch)
                    child.L = parent.L.copy()
                    child.B = parent.B.copy()
                    child.u_hat = parent.u_hat.copy()
                    child.pm = pm
                    child.u_hat[l] = bit
                    child.B[l, n] = bit
                    self._update_bits(child, l)
                    new_paths.append(child)
                paths = new_paths

        best_pm = float("inf")
        best_path = paths[0]
        crc_pass = []

        for p in paths:
            if self.crc_length > 0:
                payload = p.u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    crc_pass.append(p)
            if p.pm < best_pm:
                best_pm = p.pm
                best_path = p

        if crc_pass:
            best_path = min(crc_pass, key=lambda p: p.pm)
            best_pm = best_path.pm

        return best_path.u_hat.copy(), best_pm
