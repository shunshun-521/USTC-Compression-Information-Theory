"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _update_bits, _update_llrs, bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb ^ int(bit):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb ^ int(bit):
            reg ^= poly & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        v = 0 if llr >= 0 else 1
        if u != v:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pm = self._pm_update(path.pm, llr, 0)
                    cp = _Path(self.N, self.n)
                    cp.L[:] = path.L
                    cp.B[:] = path.B
                    cp.pm = pm
                    cp.u_hat[:] = path.u_hat
                    cp.B[l, self.n] = 0
                    cp.u_hat[l] = 0
                    _update_bits(cp.B, l, self.n)
                    new_paths.append(cp)
                else:
                    for u in (0, 1):
                        pm = self._pm_update(path.pm, llr, u)
                        cp = _Path(self.N, self.n)
                        cp.L[:] = path.L
                        cp.B[:] = path.B
                        cp.pm = pm
                        cp.u_hat[:] = path.u_hat
                        cp.B[l, self.n] = u
                        cp.u_hat[l] = u
                        _update_bits(cp.B, l, self.n)
                        new_paths.append(cp)

            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            crc_ok = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if crc_ok:
                best = min(crc_ok, key=lambda x: x.pm)

        return best.u_hat.astype(int), best.pm
