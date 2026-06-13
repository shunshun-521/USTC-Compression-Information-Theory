"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import INF, _compute_li, precompute_sc_indices


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divide(bits, poly, crc_length):
    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            poly_bits = [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]
            reg ^= np.array(poly_bits, dtype=np.int8)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    """单条译码路径。"""

    __slots__ = ("pm", "u_hat", "llrs", "s")

    def __init__(self, n, N, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.llrs = np.full((n + 1, N), -INF, dtype=np.float64)
        self.llrs[n, :] = llr_ch.copy()
        self.s = np.full((n + 1, N), -1, dtype=np.int8)

    def copy(self):
        return copy.deepcopy(self)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时深拷贝）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.n, self.N, llr_ch)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_leaf = _compute_li(0, phi, path.llrs, path.s)

                if self.frozen_bits[phi]:
                    p = path.copy()
                    p.pm = self._pm_update(p.pm, llr_leaf, 0)
                    p.u_hat[phi] = 0
                    p.llrs[0, phi] = INF
                    p.s[0, phi] = 0
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm = self._pm_update(p.pm, llr_leaf, u)
                        p.u_hat[phi] = u
                        p.llrs[0, phi] = llr_leaf
                        p.s[0, phi] = u
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
