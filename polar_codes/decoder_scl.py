"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
    _prepare_llr,
)


def _crc_process(bits, poly, crc_length):
    """LSB-first CRC 处理，返回寄存器余数。"""
    reg = 0
    for bit in bits:
        feedback = (reg ^ int(bit)) & 1
        reg >>= 1
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_process(padded, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = 0x07 if crc_length == 8 else 0x8005
    return _crc_process(bits, poly, crc_length) == 0


class _Path:
    """单条译码路径。"""

    __slots__ = ('pm', 'u_hat', 'L', 'B')

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1))
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _leaf_llr(self, path, l):
        _update_llrs(path.L, path.B, l, self.n, self.N)
        return path.L[l, self.n]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr = _prepare_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n, llr)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                llr_leaf = self._leaf_llr(path, l)

                if l in self.frozen_set:
                    penalty = 0.0 if llr_leaf >= 0 else abs(llr_leaf)
                    new_path = _Path(self.N, self.n, llr)
                    new_path.pm = path.pm + penalty
                    new_path.u_hat = path.u_hat.copy()
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr_leaf >= 0) or (bit == 1 and llr_leaf < 0) else abs(llr_leaf)
                        new_path = _Path(self.N, self.n, llr)
                        new_path.pm = path.pm + penalty
                        new_path.u_hat = path.u_hat.copy()
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths
                if crc_check(p.u_hat[info_pos], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat, best.pm
