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
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(crc_length):
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
    bits = np.asarray(bits, dtype=int)
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    """单条译码路径。"""

    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)

    def copy(self):
        p = _Path(self.N, self.n, self.L[:, 0].copy())
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制路径对象）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.sort(np.where(self.frozen_bits == 0)[0])
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_val = path.B[j - branch_size, s + 1]
                    top_bit = 0 if np.isnan(top_val) else int(top_val)
                    path.L[j, s + 1] = g_operation(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _safe_int(self, val):
        return 0 if np.isnan(val) else int(val)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        self._safe_int(path.B[j, s])
                        ^ self._safe_int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in [_bit_reversed(i, self.n) for i in range(self.N)]:
            active = []
            for path in paths:
                self._update_llrs(path, phi)
                llr_dec = path.L[phi, self.n]

                if phi in self.frozen_set:
                    path.B[phi, self.n] = 0
                    path.u_hat[phi] = 0
                    if llr_dec < 0:
                        path.pm += abs(llr_dec)
                    self._update_bits(path, phi)
                    active.append(path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.B[phi, self.n] = bit
                        new_path.u_hat[phi] = bit
                        hard = 0 if llr_dec >= 0 else 1
                        if bit != hard:
                            new_path.pm += abs(llr_dec)
                        self._update_bits(new_path, phi)
                        active.append(new_path)

            active.sort(key=lambda p: p.pm)
            paths = active[:self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
