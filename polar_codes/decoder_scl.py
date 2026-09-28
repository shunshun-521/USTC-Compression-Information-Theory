"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _align_channel_llr,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加（CRC-8: 0x07, CRC-16: 0x8005）。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ poly
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)

    def clone(self):
        cp = _Path.__new__(_Path)
        cp.L = self.L.copy()
        cp.B = self.B.copy()
        cp.pm = self.pm
        cp.u = self.u.copy()
        return cp


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径仅在写入时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            None if info_indices is None else np.asarray(info_indices, dtype=int)
        )

    def _llr_penalty(self, llr_val, u_bit):
        """路径度量增量：与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def _advance_paths_llr(self, paths, l):
        n = self.n
        for path in paths:
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size >> 1
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            path.B[j - branch_size, s + 1],
                        )

    def _update_bits(self, path, l):
        n = self.n
        if l < self.N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = _align_channel_llr(llr_ch)
        N, n = self.N, self.n
        paths = [_Path(N, n, llr_ch)]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            for p in paths:
                self._advance_paths_llr([p], l)

            new_paths = []
            for p in paths:
                llr_leaf = p.L[l, n]
                if self.frozen_bits[l]:
                    pen = self._llr_penalty(llr_leaf, 0)
                    cp = p.clone()
                    cp.pm += pen
                    cp.u[l] = 0
                    cp.B[l, n] = 0
                    self._update_bits(cp, l)
                    new_paths.append(cp)
                else:
                    for u_bit in (0, 1):
                        pen = self._llr_penalty(llr_leaf, u_bit)
                        cp = p.clone()
                        cp.pm += pen
                        cp.u[l] = u_bit
                        cp.B[l, n] = u_bit
                        self._update_bits(cp, l)
                        new_paths.append(cp)

            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                if self.info_indices is not None:
                    payload = p.u[self.info_indices]
                else:
                    payload = p.u
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x.pm)
        return best.u.copy(), best.pm
