"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    upper_llr, lower_llr, hard_decision,
    active_llr_level, active_bit_level, bit_reversed,
)
from encoder import bit_reversal_permutation


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode_bits(info_bits, crc_length=8):
    """计算 CRC 校验位"""
    poly = CRC_POLYS[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    top_bit = 1 << (crc_length - 1)

    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        if reg & top_bit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    return np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int
    )


def crc_encode(info_bits, crc_length=8):
    """信息比特 + CRC 校验位"""
    info_bits = np.asarray(info_bits, dtype=int)
    return np.concatenate([info_bits, crc_encode_bits(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    return np.array_equal(crc_encode_bits(info, crc_length), bits[-crc_length:])


class PathState:
    """单条译码路径"""

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0

    def copy(self):
        p = PathState(self.N, self.n)
        p.L[:] = self.L
        p.B[:] = self.B
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, path, l):
        n, N = self.n, self.N
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, path, l):
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        llr_tree = llr_ch[self.br].copy()
        paths = [PathState(N, n)]
        paths[0].L[:, 0] = llr_tree

        for phi in range(N):
            l = bit_reversed(phi, n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.B[l, n] = 0
                    if llr < 0:
                        new_path.pm += abs(llr)
                    candidates.append((new_path.pm, new_path))
                else:
                    for bit in [0, 1]:
                        new_path = path.copy()
                        new_path.B[l, n] = bit
                        if bit != hard_decision(llr):
                            new_path.pm += abs(llr)
                        candidates.append((new_path.pm, new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [c[1] for c in candidates[:self.list_size]]

            for path in paths:
                self._update_bits(path, l)

        # 选择最优路径
        if self.crc_length > 0:
            crc_paths = []
            for path in paths:
                u_hat = path.B[:, n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(path)
            if crc_paths:
                best = min(crc_paths, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, n].astype(int), best.pm
