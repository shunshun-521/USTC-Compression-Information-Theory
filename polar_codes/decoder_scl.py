"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _update_bits, _update_llrs, bit_reversed_index


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch

    def clone(self):
        p = _Path.__new__(_Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        from encoder import bit_reversal_permutation

        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_val = path.L[l, self.n]
                if l in self.frozen_set:
                    p = path.clone()
                    p.pm += self._pm_penalty(llr_val, 0)
                    p.B[l, self.n] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    candidates.append(p)
                else:
                    for u_val in (0, 1):
                        p = path.clone()
                        p.pm += self._pm_penalty(llr_val, u_val)
                        p.B[l, self.n] = u_val
                        _update_bits(p.B, l, self.n, self.N)
                        candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda p: p.pm)
        u_hat = best.B[:, self.n].astype(int)

        if self.crc_length > 0:
            crc_ok = []
            for path in paths:
                u = path.B[:, self.n].astype(int)
                payload = u[self.info_indices]
                if crc_check(payload, self.crc_length):
                    crc_ok.append(path)
            if crc_ok:
                best = min(crc_ok, key=lambda p: p.pm)
                u_hat = best.B[:, self.n].astype(int)

        return u_hat, best.pm
