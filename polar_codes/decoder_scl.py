"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Vangala 置换 SC
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    active_llr_level,
    active_bit_level,
)
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
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
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(expected, bits)


class Path:
    """SCL 单条路径"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)

    def copy_from(self, other):
        self.L[:] = other.L
        self.B[:] = other.B
        self.pm = other.pm
        self.u_hat[:] = other.u_hat


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
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
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_val):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_val == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size

        paths = [Path(self.N, self.n) for _ in range(L)]
        paths[0].L[:, 0] = llr_ch
        active = 1

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for pidx in range(active):
                path = paths[pidx]
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = self._pm_penalty(llr, 0)
                    path.pm += penalty
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append((path.pm, pidx, None))
                else:
                    for u_val in (0, 1):
                        penalty = self._pm_penalty(llr, u_val)
                        candidates.append((path.pm + penalty, pidx, u_val))

            candidates.sort(key=lambda x: x[0])

            new_paths = []
            for pm, src_idx, u_val in candidates[:L]:
                if u_val is None:
                    new_paths.append(paths[src_idx])
                else:
                    if len(new_paths) < len(candidates):
                        new_path = Path(self.N, self.n)
                    else:
                        new_path = paths[len(new_paths)]
                    new_path.copy_from(paths[src_idx])
                    new_path.pm = pm
                    new_path.u_hat[l] = u_val
                    new_path.B[l, self.n] = u_val
                    self._update_bits(new_path, l)
                    new_paths.append(new_path)

            paths = new_paths
            active = len(paths)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
