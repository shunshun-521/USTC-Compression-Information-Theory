"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits_pscd,
    _update_llrs_pscd,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f'Unsupported CRC length: {crc_length}')


def _crc_remainder(bits, crc_length):
    """MSB-first 按位 CRC 余数计算。"""
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]),
        crc_length,
    )
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    @staticmethod
    def _path_metric_update(pm, llr, u):
        natural = 0 if llr >= 0 else 1
        penalty = 0.0 if u == natural else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L_size = self.list_size
        br = self.br

        class Path:
            __slots__ = ('pm', 'L', 'B')

            def __init__(self):
                self.pm = 0.0
                self.L = np.zeros((N, n + 1), dtype=np.float64)
                self.B = np.zeros((N, n + 1), dtype=np.int8)
                self.L[:, 0] = llr_ch[br]

        paths = [Path()]

        for phi in range(N):
            l = br[phi]
            new_paths = []

            for path in paths:
                _update_llrs_pscd(path.L, path.B, l, n, N)
                llr = path.L[l, n]

                if self.frozen_bits[l]:
                    u = 0
                    path.B[l, n] = u
                    path.pm = self._path_metric_update(path.pm, llr, u)
                    _update_bits_pscd(path.B, l, n, N)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = Path()
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.B[l, n] = u
                        child.pm = self._path_metric_update(path.pm, llr, u)
                        _update_bits_pscd(child.B, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:L_size]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths
                if crc_check(p.B[info_idx, n], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.B[:, n].astype(int).copy(), best.pm
