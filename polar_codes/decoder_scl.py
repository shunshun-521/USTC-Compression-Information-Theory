"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bit_reversed(i, n):
    return int(bit_reversal_permutation(1 << n)[i])


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << crc_length
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & top:
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 B/L）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)
        self.phase_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.phase_order:
            for path in paths:
                self._update_llrs(path, l)

            if self.frozen_bits[l]:
                llr_val = paths[0].L[l, self.n]
                for path in paths:
                    if llr_val < 0:
                        path.pm += abs(llr_val)
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
            else:
                candidates = []
                for pidx, path in enumerate(paths):
                    llr_val = path.L[l, self.n]
                    for u in (0, 1):
                        pm = path.pm + (0.0 if (u == 0 and llr_val >= 0) or (u == 1 and llr_val < 0) else abs(llr_val))
                        candidates.append((pm, pidx, u))

                candidates.sort(key=lambda x: x[0])
                survivors = candidates[: self.list_size]

                new_paths = []
                for pm, pidx, u in survivors:
                    new_path = _Path(self.N, self.n, llr_ch)
                    new_path.pm = pm
                    new_path.L = paths[pidx].L.copy()
                    new_path.B = paths[pidx].B.copy()
                    new_path.B[l, self.n] = u
                    self._update_bits(new_path, l)
                    new_paths.append(new_path)
                paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path.B[:, self.n].astype(int)
                if self.info_indices is not None:
                    payload = u_hat[self.info_indices]
                else:
                    payload = u_hat
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int), best.pm
