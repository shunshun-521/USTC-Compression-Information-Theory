"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, _active_bit_level, _active_llr_level
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_shift(info_bits, poly, crc_length):
    """按位移寄存器计算 CRC 余数。"""
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_shift(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_shift(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _path_metric_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _compute_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l, u_val):
        path.u_hat[l] = u_val
        path.B[l, self.n] = u_val
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        init = _Path(N, n)
        init.L[:, 0] = llr_ch
        paths = [init]

        for phi in range(N):
            l = bit_reversed(phi, n)
            candidates = []

            for path in paths:
                self._compute_llr(path, l)
                llr0 = path.L[l, n]

                if self.frozen_bits[l]:
                    new_path = _Path(N, n)
                    new_path.pm = path.pm + self._path_metric_penalty(llr0, 0)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.u_hat = path.u_hat.copy()
                    self._update_bits(new_path, l, 0)
                    candidates.append((new_path.pm, new_path))
                else:
                    for u in (0, 1):
                        new_path = _Path(N, n)
                        new_path.pm = path.pm + self._path_metric_penalty(llr0, u)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat = path.u_hat.copy()
                        self._update_bits(new_path, l, u)
                        candidates.append((new_path.pm, new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [c[1] for c in candidates[: self.list_size]]

        if self.crc_length > 0:
            crc_ok = []
            for i, path in enumerate(paths):
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(i)
            best_idx = min(crc_ok, key=lambda i: paths[i].pm) if crc_ok else int(
                np.argmin([p.pm for p in paths])
            )
        else:
            best_idx = int(np.argmin([p.pm for p in paths]))

        return paths[best_idx].u_hat.copy(), paths[best_idx].pm
