"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level, _bit_reversed, _update_llrs, _update_bits


# CRC-8: 0x07, CRC-16: 0x8005
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_len) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_len) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self._br = bit_reversal_permutation(N)

    def _llr_at_bit(self, path, l):
        L, B = path.L, path.B
        _update_llrs(L, B, l, self.n)
        return float(L[l, self.n])

    def _propagate_bit(self, path, l, bit):
        path.B[l, self.n] = bit
        path.u_hat[l] = bit
        _update_bits(path.B, l, self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self._br]
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                llr = self._llr_at_bit(path, l)
                if l in self.frozen_set:
                    bit = 0
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    child = _Path(self.N, self.n, llr_ch)
                    child.L[:] = path.L
                    child.B[:] = path.B
                    child.pm = path.pm + penalty
                    child.u_hat[:] = path.u_hat
                    self._propagate_bit(child, l, bit)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
                        child = _Path(self.N, self.n, llr_ch)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.pm = path.pm + penalty
                        child.u_hat[:] = path.u_hat
                        self._propagate_bit(child, l, bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_bits_all = [p.u_hat[~self.frozen_bits] for p in paths]
            valid = [p for p, bits in zip(paths, info_bits_all) if crc_check(bits, self.crc_length)]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, K=32, seed=0):
    """L=1 的 SCL 应与 SC 一致。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(seed)
    sigma = eb_n0_to_sigma(8.0, K / N)
    scl = SCLDecoder(N, frozen, list_size=1, crc_length=0)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
