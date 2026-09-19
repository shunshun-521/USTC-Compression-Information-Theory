"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    f_operation,
    g_operation,
)
from encoder import bit_reversed_index


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005（MSB-first）
    """
    info_bits = np.asarray(info_bits, dtype=np.int32)
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
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=np.int32,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否包含正确的 CRC。"""
    bits = np.asarray(bits, dtype=np.int32)
    return np.array_equal(
        bits[-crc_length:],
        crc_encode(bits[:-crc_length], crc_length)[-crc_length:],
    )


class Path:
    """SCL 单条路径"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)

    def copy(self):
        new_path = Path.__new__(Path)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


def _update_llrs_path(path, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top_llr = path.L[j, s]
                btm_llr = path.L[j + branch_size, s]
                path.L[j, s + 1] = f_operation(top_llr, btm_llr)
            else:
                btm_llr = path.L[j, s]
                top_llr = path.L[j - branch_size, s]
                top_bit = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 通过路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError(f"N={N} must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        N, n = self.N, self.n
        paths = [Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed_index(i, n)
            candidates = []

            for path in paths:
                _update_llrs_path(path, l, n, N)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    penalty = self._path_metric_penalty(llr, 0)
                    path.pm += penalty
                    path.B[l, n] = 0
                    path.u_hat[l] = 0
                    _update_bits(path.B, l, n, N)
                    candidates.append(path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        penalty = self._path_metric_penalty(llr, u_bit)
                        new_path.pm += penalty
                        new_path.B[l, n] = u_bit
                        new_path.u_hat[l] = u_bit
                        _update_bits(new_path.B, l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_path = paths[0]
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = [
                p for p in paths
                if crc_check(p.u_hat[info_mask], self.crc_length)
            ]
            if valid:
                best_path = min(valid, key=lambda p: p.pm)

        return best_path.u_hat.copy(), best_path.pm
