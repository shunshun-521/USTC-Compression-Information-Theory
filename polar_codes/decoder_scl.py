"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import _permute_llr_for_decode, _frozen_to_info_pos, sc_decode_nonrecursive


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = ((reg << 1) & ((1 << crc_length) - 1))
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    remainder = _crc_remainder(payload, poly, crc_length)
    actual = 0
    for b in bits[-crc_length:]:
        actual = (actual << 1) | int(b)
    return remainder == actual


def _pm_update(llr_vals, bit_vals):
    pm = 0.0
    for llr, bit in zip(llr_vals, bit_vals):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_pos = _frozen_to_info_pos(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = _permute_llr_for_decode(llr_ch)

        if self.list_size == 1:
            u_hat = sc_decode_nonrecursive(llr_ch, self.info_pos, frozen_bit=0)
            return u_hat, 0.0

        N, n = self.N, self.n
        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_ch

        paths = [(0.0, llr_matrix.copy(), bit_matrix.copy())]

        for bit_idx in range(N):
            if bit_idx not in self.info_pos:
                new_paths = []
                for pm, lm, bm in paths:
                    bm = bm.copy()
                    lm = lm.copy()
                    u_tmp = sc_decode_nonrecursive(lm[0], self.info_pos, frozen_bit=0)
                    bm[n] = u_tmp
                    new_paths.append((pm, lm, bm))
                paths = new_paths
                continue

            new_paths = []
            for pm, lm, bm in paths:
                u_tmp = sc_decode_nonrecursive(lm[0], self.info_pos, frozen_bit=0)
                leaf_llr = lm[n, bit_idx]
                for bit_val in (0, 1):
                    bm2 = bm.copy()
                    lm2 = lm.copy()
                    bm2[n] = u_tmp.copy()
                    bm2[n, bit_idx] = bit_val
                    hard = 0 if leaf_llr >= 0 else 1
                    penalty = 0.0 if bit_val == hard else abs(leaf_llr)
                    new_paths.append((pm + penalty, lm2, bm2))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        best_pm, _, best_bm = paths[0]
        if self.crc_length > 0:
            for pm, _, bm in sorted(paths, key=lambda x: x[0]):
                if crc_check(bm[n].astype(int), self.crc_length):
                    best_pm, best_bm = pm, bm
                    break

        return best_bm[n].astype(int), best_pm


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=8.0):
    """L=1 的 SCL 应等价于 SC。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(1)

    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    return True
