"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    sc_decode_nonrecursive,
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
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _phase_llr(phi, layer, llr_ch, bits, n):
    if layer == n:
        return llr_ch[phi]
    step = 1 << layer
    if (phi >> layer) & 1 == 0:
        return f_operation(
            _phase_llr(phi, layer + 1, llr_ch, bits, n),
            _phase_llr(phi | (1 << layer), layer + 1, llr_ch, bits, n),
        )
    left = phi - step
    return g_operation(
        _phase_llr(left, layer + 1, llr_ch, bits, n),
        _phase_llr(phi, layer + 1, llr_ch, bits, n),
        bits[layer][left],
    )


def _propagate_bits(phi, bits, bit_layer_vec):
    for layer in bit_layer_vec[phi]:
        step = 1 << layer
        beta = (phi // (2 * step)) * (2 * step)
        i = phi % step
        bits[layer + 1][beta + i] = bits[layer][beta + i] ^ bits[layer][beta + step + i]
        bits[layer + 1][beta + step + i] = bits[layer][beta + step + i]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        _, _, self.bit_layer_vec = precompute_sc_indices(N)

    def _metric_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        if L == 1:
            return sc_decode_nonrecursive(llr_ch, self.frozen_bits), 0.0

        paths = [{"pm": 0.0, "u": np.zeros(N, dtype=int), "bits": np.zeros((n + 1, N), dtype=int)}]

        for phi in range(N):
            candidates = []
            for path in paths:
                llr_val = _phase_llr(phi, 0, llr_ch, path["bits"], n)
                choices = [0] if self.frozen_bits[phi] else [0, 1]
                for u_bit in choices:
                    bits_new = path["bits"].copy()
                    u_new = path["u"].copy()
                    u_new[phi] = u_bit
                    bits_new[0][phi] = u_bit
                    _propagate_bits(phi, bits_new, self.bit_layer_vec)
                    pm = path["pm"] + self._metric_penalty(llr_val, u_bit)
                    candidates.append({"pm": pm, "u": u_new, "bits": bits_new})

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:L]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p["pm"]) if valid else min(paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u"].copy(), best["pm"]


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=5.0):
    from construction import ga_construction
    from encoder import polar_encode_natural
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode_natural(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)

    print("SCL L=1 == SC verification passed")


if __name__ == "__main__":
    verify_scl_equals_sc()
