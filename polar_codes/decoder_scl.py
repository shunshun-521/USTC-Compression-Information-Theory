"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _prepare_channel_llrs,
    _frozen_indices_from_mask,
)


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_core(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = ((reg << 1) & ((1 << crc_length) - 1))
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_core(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=np.int32)

    def copy(self):
        p = _Path(len(self.L), int(np.log2(len(self.L))), self.L[:, 0].copy())
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = _frozen_indices_from_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

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
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
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

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr = _prepare_channel_llrs(llr_ch)
        paths = [_Path(self.N, self.n, llr)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    penalty = self._path_metric_penalty(llr_bit, 0)
                    path.pm += penalty
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._path_metric_penalty(llr_bit, bit)
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
