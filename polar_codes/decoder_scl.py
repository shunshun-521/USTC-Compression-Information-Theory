"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        start_s = self.n - _active_llr_level(l, self.n)
        for s in range(start_s, self.n):
            block = 1 << (s + 1)
            branch = block >> 1
            for j in range(l, self.N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch, s], L[j, s], B[j - branch, s + 1]
                    )

    def _update_bits(self, path, l, u):
        path.u_hat[l] = u
        path.B[l, self.n] = u
        if l < self.N // 2:
            return
        B = path.B
        stop_s = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, stop_s, -1):
            block = 1 << s
            branch = block >> 1
            for j in range(l, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        L_size = self.list_size

        root = _Path(self.N, n)
        root.L[:, 0] = llr_ch
        active = [root]

        for l in self.decode_order:
            candidates = []
            for path in active:
                self._update_llrs(path, l)
                llr_leaf = path.L[l, n]
                if self.frozen_bits[l]:
                    candidates.append((path.pm, path, 0))
                else:
                    for u in (0, 1):
                        candidates.append((_pm_update(path.pm, llr_leaf, u), path, u))

            candidates.sort(key=lambda x: x[0])
            next_active = []
            for pm, parent, u in candidates[:L_size]:
                child = _Path(self.N, n)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                self._update_bits(child, l, u)
                next_active.append(child)
            active = next_active

        crc_ok = []
        for path in active:
            if self.crc_length > 0:
                info_bits = path.u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(path)

        if crc_ok:
            best = min(crc_ok, key=lambda p: p.pm)
            return best.u_hat.copy(), best.pm

        best = min(active, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, seed=1):
    """L=1 的 SCL 应与 SC 等价（无 CRC）"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from decoder_sc import sc_decode
    from encoder import polar_encode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(4.0, K / N)
    rng = np.random.default_rng(seed)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("SCL(L=1) 与 SC 不一致")
