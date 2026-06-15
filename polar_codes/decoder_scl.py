"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation_exact,
    g_operation,
    precompute_sc_indices,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg


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
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class Path:
    """单条 SCL 路径。"""

    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n, llr_ch=None):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)
        if llr_ch is not None:
            self.L[:, 0] = llr_ch

    def copy(self):
        new = Path(self.L.shape[0], self.L.shape[1] - 1)
        new.pm = self.pm
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.u_hat = self.u_hat.copy()
        return new


def _update_llrs(path, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = f_operation_exact(path.L[j, s], path.L[j + branch_size, s])
            else:
                top = path.L[j - branch_size, s]
                btm = path.L[j, s]
                top_bit = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = g_operation(top, btm, top_bit)


def _update_bits(path, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, _ = precompute_sc_indices(N)

    @staticmethod
    def _metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            for path in paths:
                _update_llrs(path, l, self.n, self.N)

            llr_leaf = paths[0].L[l, self.n]
            new_paths = []

            if self.frozen_bits[l]:
                for path in paths:
                    path.pm += self._metric_penalty(llr_leaf, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(path, l, self.n, self.N)
                    new_paths.append(path)
            else:
                for path in paths:
                    llr_use = path.L[l, self.n]
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm += self._metric_penalty(llr_use, bit)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        _update_bits(child, l, self.n, self.N)
                        new_paths.append(child)
                new_paths.sort(key=lambda p: p.pm)
                paths = new_paths[:self.list_size]
                continue

            paths = new_paths

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths
                if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]
        return best.u_hat.copy(), best.pm
