"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed_int,
    _update_llrs,
    _update_bits,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加到信息比特后（LFSR，多项式 0x07 / 0x8005）"""
    r = crc_length
    poly = _crc_poly(r)
    state = 0
    for bit in info_bits:
        fb = (state >> (r - 1)) ^ int(bit)
        state = (state << 1) & ((1 << r) - 1)
        if fb:
            state ^= poly
    crc_bits = np.array([(state >> (r - 1 - i)) & 1 for i in range(r)], dtype=np.int8)
    return np.concatenate([info_bits.astype(np.int8), crc_bits])


def crc_check(bits, crc_length=8):
    r = crc_length
    poly = _crc_poly(r)
    state = 0
    for bit in bits:
        fb = (state >> (r - 1)) ^ int(bit)
        state = (state << 1) & ((1 << r) - 1)
        if fb:
            state ^= poly
    return state == 0


class _Path:
    __slots__ = ("L", "B", "pm", "n", "N")

    def __init__(self, N, n, llr=None, parent=None):
        self.N = N
        self.n = n
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
            if llr is not None:
                self.L[:, 0] = llr
            self.pm = 0.0
        else:
            self.L = parent.L.copy()
            self.B = parent.B.copy()
            self.pm = parent.pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr=llr_ch)]

        for i in range(self.N):
            l = bit_reversed_int(i, self.n)
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = 0.0 if llr >= 0 else abs(llr)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = _Path(self.N, self.n, parent=path)
                        pen = 0.0 if (llr >= 0 and u == 0) or (llr < 0 and u == 1) else abs(llr)
                        child.pm += pen
                        child.B[l, self.n] = u
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L_size]

        crc_ok = []
        for p in paths:
            u_hat = p.B[:, self.n].astype(np.int8)
            info_bits = u_hat[self.info_indices]
            if self.crc_length > 0:
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(p)
            else:
                crc_ok.append(p)

        best = min(crc_ok if crc_ok else paths, key=lambda p: p.pm)
        u_hat = best.B[:, self.n].astype(np.int8)
        return u_hat, best.pm
