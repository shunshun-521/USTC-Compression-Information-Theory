"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _frozen_set,
)


def _crc_generator_bits(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)  # x^8+x^2+x+1
    if crc_length == 16:
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int)
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    gen = _crc_generator_bits(crc_length)
    msg = np.asarray(bits, dtype=int).tolist()
    n = len(gen) - 1
    for i in range(len(msg) - n):
        if msg[i]:
            for j in range(len(gen)):
                if i + j < len(msg):
                    msg[i + j] ^= gen[j]
    return np.array(msg[-n:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    rem = _crc_remainder(bits, crc_length)
    return np.all(rem == 0)


def _path_metric_update(pm, llr, u_bit):
    hard = 0 if llr >= 0 else 1
    if u_bit != hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.active = True

    def copy(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.active = True
        return p


class SCLDecoder:
    """SCL 译码器（路径复制实现，适用于中等列表大小）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen = _frozen_set(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(self.frozen_bits == 0)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
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
        llr_core = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [_Path(self.N, self.n, llr_core)]

        for l in self.br:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen:
                    pm = _path_metric_update(path.pm, llr, 0)
                    new_path = path.copy()
                    new_path.pm = pm
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        pm = _path_metric_update(path.pm, llr, u_bit)
                        new_path = path.copy()
                        new_path.pm = pm
                        new_path.B[l, self.n] = u_bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = None
        if self.crc_length > 0:
            for path in sorted(paths, key=lambda p: p.pm):
                u_hat = path.B[:, self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    best = path
                    break
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm
