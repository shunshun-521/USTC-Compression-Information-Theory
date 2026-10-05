"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _update_bits, _update_llrs
from encoder import bit_reversed as br_index


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_update(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    reg ^= int(bit) << (crc_length - 1)
    for _ in range(crc_length):
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg = _crc_update(reg, b, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg = _crc_update(reg, b, poly, crc_length)
    return reg == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u_hat = {}


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, path):
        new_p = _Path(self.N, self.n, np.zeros(self.N))
        new_p.pm = path.pm
        new_p.L = path.L.copy()
        new_p.B = path.B.copy()
        new_p.u_hat = dict(path.u_hat)
        return new_p

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = br_index(i, self.n)
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr0 = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr0, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr0, u)
                        new_path.u_hat[l] = u
                        new_path.B[l, self.n] = u
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        best_u = None
        for path in paths:
            u_hat = np.zeros(self.N, dtype=int)
            for idx, bit in path.u_hat.items():
                u_hat[idx] = bit
            pm = path.pm
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or pm < best_crc[1]:
                        best_crc = (u_hat, pm)
            if best_pm is None or pm < best_pm:
                best_pm = pm
                best_u = u_hat

        if best_crc is not None:
            return best_crc[0], best_crc[1]
        return best_u, best_pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(6.0, 0.5)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 must match SC"
    print("SCL L=1 equivalence test passed")
