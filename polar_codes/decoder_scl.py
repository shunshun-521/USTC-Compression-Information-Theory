"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    f_operation,
    g_operation,
    _permute_channel_llrs,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = _permute_channel_llrs(llr_ch, self.N)
        n = self.n
        N = self.N

        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
            "u_hat": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr

        def update_llrs(path, l):
            for s in range(n - _active_llr_level(l, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        path["L"][j, s + 1] = f_operation(
                            path["L"][j, s], path["L"][j + branch_size, s]
                        )
                    else:
                        path["L"][j, s + 1] = g_operation(
                            path["L"][j - branch_size, s],
                            path["L"][j, s],
                            path["B"][j - branch_size, s + 1],
                        )

        def update_bits(path, l):
            if l < N // 2:
                return
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path["B"][j - branch_size, s - 1] = (
                            path["B"][j, s] ^ path["B"][j - branch_size, s]
                        )
                        path["B"][j, s - 1] = path["B"][j, s]

        for idx in range(N):
            l = _bit_reversed_index(idx, n)
            candidates = []

            for path in paths:
                update_llrs(path, l)
                llr0 = path["L"][l, n]

                if l in self.frozen_set:
                    new_path = {
                        "pm": path["pm"] + self._path_metric_penalty(llr0, 0),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["B"][l, n] = 0
                    new_path["u_hat"][l] = 0
                    update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            "pm": path["pm"] + self._path_metric_penalty(llr0, bit),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["B"][l, n] = bit
                        new_path["u_hat"][l] = bit
                        update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            if crc_pass:
                paths = crc_pass

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]


def verify_scl_equals_sc(N=64, eb_n0_db=10.0, num_trials=50):
    """验证 L=1 的 SCL 等价于 SC"""
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from decoder_sc import sc_decode
    from encoder import polar_encode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    return True
