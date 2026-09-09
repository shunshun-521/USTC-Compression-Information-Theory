"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation, bit_reversed,
    active_llr_level, active_bit_level,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f'Unsupported CRC length: {crc_length}')


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg == 0


class Path:
    """单条译码路径"""
    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n, llr_ch=None):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch

    def clone(self):
        p = Path(len(self.L), self.L.shape[1] - 1)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def _llr_update(self, path, l_idx):
        for s in range(self.n - active_llr_level(l_idx, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )
        return path.L[l_idx, self.n]

    def _bit_propagate(self, path, l_idx):
        if l_idx < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l_idx, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l_idx = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                llr_root = self._llr_update(path, l_idx)

                if l_idx in self.frozen_set:
                    new_path = path.clone()
                    penalty = abs(llr_root) if llr_root < 0 else 0.0
                    new_path.pm += penalty
                    new_path.u_hat[l_idx] = 0
                    new_path.B[l_idx, self.n] = 0
                    self._bit_propagate(new_path, l_idx)
                    candidates.append(new_path)
                else:
                    bit_decision = 0 if llr_root >= 0 else 1
                    for u_val in (0, 1):
                        new_path = path.clone()
                        penalty = 0.0 if u_val == bit_decision else abs(llr_root)
                        new_path.pm += penalty
                        new_path.u_hat[l_idx] = u_val
                        new_path.B[l_idx, self.n] = u_val
                        self._bit_propagate(new_path, l_idx)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        best = None
        if self.crc_length > 0:
            crc_pass = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
