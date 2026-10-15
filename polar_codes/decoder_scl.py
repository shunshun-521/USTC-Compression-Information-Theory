"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    bit_reversal_permutation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_bits_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    rem = _crc_bits_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int64
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int64)
    return _crc_bits_remainder(bits, crc_length) == 0


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_init):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_init
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        n = self.L.shape[1] - 1
        p = object.__new__(_Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SCD + 路径列表）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr0 = path.L[l, self.n]
                if l in self.frozen_set:
                    cp = path.copy()
                    cp.pm = _pm_update(path.pm, llr0, 0)
                    cp.B[l, self.n] = 0
                    cp.u_hat[l] = 0
                    _update_bits(cp.B, l, self.n)
                    new_paths.append(cp)
                else:
                    for u in (0, 1):
                        cp = path.copy()
                        cp.pm = _pm_update(path.pm, llr0, u)
                        cp.B[l, self.n] = u
                        cp.u_hat[l] = u
                        _update_bits(cp.B, l, self.n)
                        new_paths.append(cp)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = None
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            for path in paths:
                if crc_check(path.u_hat[info_idx], self.crc_length):
                    if best is None or path.pm < best.pm:
                        best = path
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
