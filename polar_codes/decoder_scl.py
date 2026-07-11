"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int64,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int64)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int64)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == hard else abs(llr_val)

    def _update_llrs(self, path, l_idx):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l_idx, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l_idx):
        if l_idx < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l_idx, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L = self.list_size

        paths = [_Path(N, n) for _ in range(L)]
        paths[0].L[:, 0] = llr_ch.copy()
        active_paths = [paths[0]]

        for phi in range(N):
            l_idx = _bit_reversed(phi, n)
            candidates = []

            for path in active_paths:
                self._update_llrs(path, l_idx)
                llr_val = path.L[l_idx, n]

                if self.frozen_bits[l_idx]:
                    pm = path.pm + self._path_metric_penalty(llr_val, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u in (0, 1):
                        pm = path.pm + self._path_metric_penalty(llr_val, u)
                        candidates.append((pm, path, u))

            candidates.sort(key=lambda x: x[0])
            survivors = candidates[:L]

            new_active = []
            for new_pm, parent, u_bit in survivors:
                if new_active and new_active[-1][1] is parent:
                    child = new_active[-1][0]
                else:
                    child = copy.copy(parent)
                    child.L = parent.L.copy()
                    child.B = parent.B.copy()
                    child.u_hat = parent.u_hat.copy()
                child.pm = new_pm
                child.B[l_idx, n] = u_bit
                child.u_hat[l_idx] = u_bit
                self._update_bits(child, l_idx)
                new_active.append((child, parent))

            active_paths = [p for p, _ in new_active]

        if self.crc_length > 0:
            valid = [
                p for p in active_paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else active_paths, key=lambda p: p.pm)
        else:
            best = min(active_paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, K=32, seed=1):
    """单路径 SCL 应等价于 SC"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from decoder_sc import sc_decode
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(6.0, K / N)
    rng = np.random.default_rng(seed)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int64)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True
