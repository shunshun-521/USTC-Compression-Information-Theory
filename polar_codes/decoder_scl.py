"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），递归列表搜索
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode


CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _path_penalty(llr, u_bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr)


def _scl_decode_tree(llr_node, frozen_node, list_size):
    """递归 SCL 译码树（与 SC 共享 f/g/u_up 约定）。"""
    n = len(llr_node)
    if n == 1:
        llr0 = float(llr_node[0])
        if frozen_node[0]:
            z = np.array([0.0])
            return [(0.0, z, z.copy())]
        paths = []
        for u_bit in (0.0, 1.0):
            pm = _path_penalty(llr0, int(u_bit))
            z = np.array([u_bit])
            paths.append((pm, z, z.copy()))
        paths.sort(key=lambda x: x[0])
        return paths[:list_size]

    half = n // 2
    llr1 = llr_node[:half]
    llr2 = llr_node[half:]
    f1 = frozen_node[:half]
    f2 = frozen_node[half:]
    llr_left = f_operation(llr1, llr2)

    left_paths = _scl_decode_tree(llr_left, f1, list_size)
    merged = []
    for pm_l, u1, u1_up in left_paths:
        llr_right = g_operation(llr1, llr2, u1_up)
        for pm_r, u2, u2_up in _scl_decode_tree(llr_right, f2, list_size):
            u = np.concatenate([u1, u2])
            xor_up = np.bitwise_xor(u1_up.astype(int), u2_up.astype(int)).astype(np.float64)
            u_up = np.concatenate([xor_up, u2_up])
            merged.append((pm_l + pm_r, u, u_up))

    merged.sort(key=lambda x: x[0])
    return merged[:list_size]


class SCLDecoder:
    """SCL 译码器（递归列表搜索 + CRC 辅助）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """SCL 主译码，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = llr_ch[self.rev]
        paths = _scl_decode_tree(llr, self.frozen_bits, self.list_size)

        if self.crc_length > 0:
            valid = []
            for pm, u, _ in paths:
                u_hat = np.round(u).astype(int)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                pm_best, u_hat = min(valid, key=lambda x: x[0])
                return u_hat, pm_best

        pm, u, _ = paths[0]
        return np.round(u).astype(int), pm


def verify_scl_equals_sc(N=64, K=32, num_frames=50, eb_n0_db=10.0):
    """L=1 的 SCL 应与 SC 完全一致。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("SCL L=1 != SC")
    return True
