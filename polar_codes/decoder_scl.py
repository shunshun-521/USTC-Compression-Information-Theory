"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy
import numpy as np
from decoder_sc import (
    sc_decode_nonrecursive,
    sc_decode_step,
    path_metric_update,
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


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        if L == 1:
            return sc_decode_nonrecursive(llr_ch, self.frozen_bits), 0.0

        llr0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr0[0] = llr_ch

        paths = [(0.0, llr0, bit0)]
        split_positions = list(self.info_indices)

        for split_pos in split_positions:
            new_paths = []
            for pm, llr_m, bit_m in paths:
                llr_t, bit_t, leaf_llr = sc_decode_step(
                    llr_m.copy(), bit_m.copy(), self.frozen_bits, split_pos
                )
                for u_bit in (0, 1):
                    bit_c = bit_t.copy()
                    bit_c[n][split_pos] = u_bit
                    pm_new = pm + path_metric_update(leaf_llr, u_bit)
                    new_paths.append((pm_new, llr_t.copy(), bit_c))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[:L]

        best_pm, llr_f, bit_f = min(paths, key=lambda x: x[0])
        u_hat = sc_decode_nonrecursive(llr_ch, self.frozen_bits)
        # 完成剩余译码：从最后状态继续
        if not np.all(~np.isnan(bit_f[n])):
            u_hat = np.nan_to_num(bit_f[n], nan=0).astype(int)
        else:
            _, bit_f, _ = sc_decode_step(llr_f, bit_f, self.frozen_bits, N - 1)
            u_hat = np.nan_to_num(bit_f[n], nan=0).astype(int)

        u_hat[self.frozen_bits] = 0

        if self.crc_length > 0:
            if not crc_check(u_hat[self.info_indices], self.crc_length):
                for pm, _, bit_m in sorted(paths, key=lambda x: x[0]):
                    uh = np.nan_to_num(bit_m[n], nan=0).astype(int)
                    if crc_check(uh[self.info_indices], self.crc_length):
                        u_hat = uh
                        best_pm = pm
                        break

        return u_hat, best_pm


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=5.0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode, channel_llr_for_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = channel_llr_for_decode(compute_llr(y, sigma))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)

    print("SCL L=1 == SC verification passed")


if __name__ == "__main__":
    verify_scl_equals_sc()
